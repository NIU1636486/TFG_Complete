# Copyright (c) 2022 Huawei Technologies Co., Ltd.
# ... (license header) ...

"""
Tiled image inpainting using RePaint model.
Divides large images into 256x256 blocks, processes each block separately.
"""

import os
import argparse
import torch as th
import torch.nn.functional as F
from PIL import Image
import numpy as np
import time
import math
import conf_mgt
from utils import yamlread
from guided_diffusion import dist_util

# ... (Workaround and imports) ...
try:
    import ctypes
    libgcc_s = ctypes.CDLL('libgcc_s.so.1')
except:
    pass

from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    select_args,
)


def load_image_full_res(image_path):
    """Load image at full resolution."""
    image = Image.open(image_path).convert('RGB')
    return np.array(image)

def load_mask_full_res(mask_path):
    """Load mask at full resolution."""
    mask = Image.open(mask_path).convert('L')
    return np.array(mask)

def divide_image_into_blocks(image, mask, block_size=256, overlap=0):
    """
    Divide image and mask into 256x256 blocks.
    For blocks at the right and bottom edges, the start coordinate is shifted
    to ensure the block is always block_size x block_size.
    """
    h, w = image.shape[:2]
    blocks = []
    
    step = block_size - overlap
    
    rows = math.ceil(h / step) if h > block_size else 1
    cols = math.ceil(w / step) if w > block_size else 1
    
    print(f"Image size: {w}x{h}")
    print(f"Block size: {block_size}x{block_size}")
    print(f"Overlap: {overlap} pixels")
    print(f"Grid: {cols}x{rows} blocks")
    print(f"Total blocks: {rows * cols}")
    
    for row in range(rows):
        for col in range(cols):
            # Calculate the nominal top-left corner for this grid cell
            start_y_nominal = row * step
            start_x_nominal = col * step
            
            # --- NEW LOGIC: Adjust start coordinates for edge blocks ---
            # If a block would extend past the image, shift its start point back
            # so that it ends exactly at the image boundary.
            
            start_y = start_y_nominal
            if start_y + block_size > h:
                start_y = h - block_size

            start_x = start_x_nominal
            if start_x + block_size > w:
                start_x = w - block_size

            # Ensure coordinates are not negative (for images smaller than block_size)
            start_y = max(0, start_y)
            start_x = max(0, start_x)

            # The slice is now guaranteed to be block_size x block_size
            end_y = start_y + block_size
            end_x = start_x + block_size
            
            img_block = image[start_y:end_y, start_x:end_x]
            mask_block = mask[start_y:end_y, start_x:end_x]

            # For the info file, we need two sets of coordinates:
            # 1. The actual slice coordinates (start_x, start_y) for placement.
            # 2. The nominal end coordinates for calculating the full image size later.
            nominal_end_y = min(start_y_nominal + block_size, h)
            nominal_end_x = min(start_x_nominal + block_size, w)

            position = {
                'row': row,
                'col': col,
                'start_y': start_y,       # Actual start Y of the 256x256 slice
                'start_x': start_x,       # Actual start X of the 256x256 slice
                'end_y': nominal_end_y,   # Logical end Y of the grid cell
                'end_x': nominal_end_x,   # Logical end X of the grid cell
                'original_h': block_size, # The block is now always full size
                'original_w': block_size  # The block is now always full size
            }
            
            blocks.append((img_block, mask_block, position))
    
    return blocks

# ... (preprocess_block, toU8, save_image functions are unchanged) ...

def preprocess_block(img_block, mask_block):
    """Preprocess image and mask blocks for the model."""
    # Normalize image to [-1, 1]
    img_normalized = img_block.astype(np.float32) / 127.5 - 1.0
    img_normalized = img_normalized.transpose(2, 0, 1)  # HWC to CHW
    
    # Normalize mask to [0, 1]
    mask_normalized = mask_block.astype(np.float32) / 255.0
    
    return img_normalized, mask_normalized

def toU8(sample):
    """Convert tensor to uint8 numpy array."""
    if sample is None:
        return sample
    
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    sample = sample.detach().cpu().numpy()
    return sample

def save_image(image_array, path):
    """Save image array as PIL Image."""
    if image_array is None:
        return
    
    # Handle different tensor shapes
    if len(image_array.shape) == 4:
        image_array = image_array[0]  # Remove batch dimension
    
    # Handle single channel images (masks)
    if len(image_array.shape) == 3 and image_array.shape[2] == 1:
        image_array = image_array[:, :, 0]  # Remove channel dimension
    
    # Handle grayscale images
    if len(image_array.shape) == 2:
        image = Image.fromarray(image_array, mode='L')
    else:
        image = Image.fromarray(image_array)
    
    image.save(path)


def save_block_info(blocks, output_dir, base_name):
    """Save block information to a text file."""
    info_path = os.path.join(output_dir, f"{base_name}_block_info.txt")
    with open(info_path, 'w') as f:
        f.write(f"Total blocks: {len(blocks)}\n")
        f.write("Block information:\n")
        f.write("Format: block_id, row, col, start_x, start_y, end_x, end_y, original_w, original_h\n")
        
        for i, (_, _, pos) in enumerate(blocks):
            f.write(f"{i:04d}, {pos['row']}, {pos['col']}, "
                   f"{pos['start_x']}, {pos['start_y']}, "
                   f"{pos['end_x']}, {pos['end_y']}, "
                   f"{pos['original_w']}, {pos['original_h']}\n")

def main(conf: conf_mgt.Default_Conf, args):
    print("Start tiled image inpainting:", conf['name'])
    
    device = dist_util.dev(conf.get('device'))
    
    # Load model
    model, diffusion = create_model_and_diffusion(
        **select_args(conf, model_and_diffusion_defaults().keys()), conf=conf
    )
    model.load_state_dict(
        dist_util.load_state_dict(os.path.expanduser(
            conf.model_path), map_location="cpu")
    )
    model.to(device)
    if conf.use_fp16:
        model.convert_to_fp16()
    model.eval()
    
    show_progress = conf.show_progress
    
    # Load classifier if specified
    if conf.classifier_scale > 0 and conf.classifier_path:
        print("loading classifier...")
        classifier = create_classifier(
            **select_args(conf, classifier_defaults().keys()))
        classifier.load_state_dict(
            dist_util.load_state_dict(os.path.expanduser(
                conf.classifier_path), map_location="cpu")
        )
        
        classifier.to(device)
        if conf.classifier_use_fp16:
            classifier.convert_to_fp16()
        classifier.eval()
        
        def cond_fn(x, t, y=None, gt=None, **kwargs):
            assert y is not None
            with th.enable_grad():
                x_in = x.detach().requires_grad_(True)
                logits = classifier(x_in, t)
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs[range(len(logits)), y.view(-1)]
                return th.autograd.grad(selected.sum(), x_in)[0] * conf.classifier_scale
    else:
        cond_fn = None
    
    def model_fn(x, t, y=None, gt=None, **kwargs):
        assert y is not None
        return model(x, t, y if conf.class_cond else None, gt=gt)
    
    # Load input images at full resolution
    print("Loading input images...")
    input_img = load_image_full_res(args.input_img)
    input_mask = load_mask_full_res(args.input_mask)
    input_gt = load_image_full_res(args.input_gt)
    
    print("Image is large, dividing into blocks...")
    # Divide into blocks using the new logic
    blocks = divide_image_into_blocks(input_img, input_mask, 
                                    block_size=256, 
                                    overlap=args.overlap)
    
    # Create output directories
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'blocks_input'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'blocks_mask'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'blocks_gt'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'blocks_inpainted'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'blocks_masked'), exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(args.input_img))[0]
    
    # Save block information
    save_block_info(blocks, output_dir, base_name)
    
    print(f"Processing {len(blocks)} blocks...")
    
    # Process each block
    for block_idx, (img_block, mask_block, position) in enumerate(blocks):
        print(f"\nProcessing block {block_idx + 1}/{len(blocks)} "
              f"(row {position['row']}, col {position['col']})")
        
        # Save input block and mask for reference
        block_name = f"{base_name}_block_{block_idx:04d}_r{position['row']:02d}_c{position['col']:02d}"
        
        Image.fromarray(img_block).save(
            os.path.join(output_dir, 'blocks_input', f"{block_name}_input.png"))
        Image.fromarray(mask_block).save(
            os.path.join(output_dir, 'blocks_mask', f"{block_name}_mask.png"))
        
        # Preprocess block
        img_normalized, mask_normalized = preprocess_block(img_block, mask_block)
        
        # Convert to tensors
        img_tensor = th.from_numpy(img_normalized).unsqueeze(0).to(device)
        mask_tensor = th.from_numpy(mask_normalized).unsqueeze(0).unsqueeze(0).to(device)
        gt_tensor = img_tensor.clone()
        
        # Prepare model kwargs
        model_kwargs = {}
        model_kwargs["gt"] = gt_tensor
        model_kwargs['gt_keep_mask'] = mask_tensor
        
        batch_size = 1
        
        if conf.cond_y is not None:
            classes = th.ones(batch_size, dtype=th.long, device=device)
            model_kwargs["y"] = classes * conf.cond_y
        else:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(batch_size,), device=device
            )
            model_kwargs["y"] = classes
        
        sample_fn = (
            diffusion.p_sample_loop if not conf.use_ddim else diffusion.ddim_sample_loop
        )
        
        start_time = time.time()
        result = sample_fn(
            model_fn,
            (batch_size, 3, 256, 256),
            clip_denoised=conf.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=device,
            progress=show_progress,
            return_all=True,
            conf=conf
        )
        end_time = time.time()
        
        print(f"Block {block_idx + 1} completed in {end_time - start_time:.2f} seconds")
        
        srs = toU8(result['sample'])
        lrs = toU8(result.get('gt') * model_kwargs.get('gt_keep_mask') + (-1) *
                   th.ones_like(result.get('gt')) * (1 - model_kwargs.get('gt_keep_mask')))
        
        save_image(srs, os.path.join(output_dir, 'blocks_inpainted', f"{block_name}_inpainted.png"))
        save_image(lrs, os.path.join(output_dir, 'blocks_masked', f"{block_name}_masked.png"))
    
    print(f"\nAll blocks processed!")
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tiled image inpainting with RePaint")
    parser.add_argument('--conf_path', type=str, required=True, 
                        help='Path to configuration file')
    parser.add_argument('--input-img', type=str, required=True,
                        help='Path to input image to be inpainted')
    parser.add_argument('--input-mask', type=str, required=True,
                        help='Path to mask image (black areas will be inpainted, white areas will be kept)')
    # Removed --input-gt as it's not used in the main loop logic
    parser.add_argument('--output-dir', type=str, default='./output',
                        help='Directory to save output images (default: ./output)')
    parser.add_argument('--overlap', type=int, default=32,
                        help='Overlap between blocks in pixels (default: 32)')
    
    args = parser.parse_args()
    
    conf_arg = conf_mgt.conf_base.Default_Conf()
    conf_arg.update(yamlread(args.conf_path))
    
    # A dummy GT is needed for the original script structure, let's use the input image
    args.input_gt = args.input_img
    main(conf_arg, args)