# Copyright (c) 2022 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This repository was forked from https://github.com/openai/guided-diffusion, which is under the MIT license

"""
Single image inpainting using RePaint model.
Takes a single image, mask, and ground truth as input arguments.
"""

import os
import argparse
import torch as th
import torch.nn.functional as F
from PIL import Image
import numpy as np
import time
import conf_mgt
from utils import yamlread
from guided_diffusion import dist_util

# Workaround
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

def load_image(image_path, size=256):
    """Load and preprocess an image."""
    image = Image.open(image_path).convert('RGB')
    image = image.resize((size, size), Image.LANCZOS)
    image = np.array(image).astype(np.float32)
    image = image / 127.5 - 1.0  # Normalize to [-1, 1]
    image = image.transpose(2, 0, 1)  # HWC to CHW
    return image

def load_mask(mask_path, size=256):
    """Load and preprocess a mask."""
    mask = Image.open(mask_path).convert('L')  # Convert to grayscale
    mask = mask.resize((size, size), Image.NEAREST)
    mask = np.array(mask).astype(np.float32)
    mask = mask / 255.0  # Normalize to [0, 1]
    # Keep mask as is - white areas (1) are kept, black areas (0) are inpainted
    return mask

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

def main(conf: conf_mgt.Default_Conf, args):
    print("Start single image inpainting:", conf['name'])
    
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
    
    # Load input images
    print("Loading input images...")
    input_img = load_image(args.input_img, conf.image_size)
    input_mask = load_mask(args.input_mask, conf.image_size)
    input_gt = load_image(args.input_gt, conf.image_size)
    
    # Convert to tensors and add batch dimension
    input_img_tensor = th.from_numpy(input_img).unsqueeze(0).to(device)
    input_mask_tensor = th.from_numpy(input_mask).unsqueeze(0).unsqueeze(0).to(device)
    input_gt_tensor = th.from_numpy(input_gt).unsqueeze(0).to(device)
    
    # Prepare model kwargs
    model_kwargs = {}
    model_kwargs["gt"] = input_gt_tensor
    model_kwargs['gt_keep_mask'] = input_mask_tensor
    
    batch_size = 1
    
    # Set up class conditioning
    if conf.cond_y is not None:
        classes = th.ones(batch_size, dtype=th.long, device=device)
        model_kwargs["y"] = classes * conf.cond_y
    else:
        classes = th.randint(
            low=0, high=NUM_CLASSES, size=(batch_size,), device=device
        )
        model_kwargs["y"] = classes
    
    # Choose sampling function
    sample_fn = (
        diffusion.p_sample_loop if not conf.use_ddim else diffusion.ddim_sample_loop
    )
    
    print("Starting inpainting...")
    start_time = time.time()
    
    # Perform inpainting
    result = sample_fn(
        model_fn,
        (batch_size, 3, conf.image_size, conf.image_size),
        clip_denoised=conf.clip_denoised,
        model_kwargs=model_kwargs,
        cond_fn=cond_fn,
        device=device,
        progress=show_progress,
        return_all=True,
        conf=conf
    )
    
    end_time = time.time()
    print(f"Inpainting completed in {end_time - start_time:.2f} seconds")
    
    # Process results
    srs = toU8(result['sample'])
    gts = toU8(result['gt'])
    lrs = toU8(result.get('gt') * model_kwargs.get('gt_keep_mask') + (-1) *
               th.ones_like(result.get('gt')) * (1 - model_kwargs.get('gt_keep_mask')))
    
    # Handle mask visualization properly
    mask_tensor = model_kwargs.get('gt_keep_mask')
    if mask_tensor is not None:
        # Convert mask to proper format for visualization
        mask_vis = ((mask_tensor + 1) * 127.5).clamp(0, 255).to(th.uint8)
        mask_vis = mask_vis.squeeze()  # Remove extra dimensions
        if len(mask_vis.shape) == 3:
            mask_vis = mask_vis.permute(1, 2, 0)
        mask_vis = mask_vis.detach().cpu().numpy()
        gt_keep_masks = mask_vis
    else:
        gt_keep_masks = None
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    base_name = os.path.splitext(os.path.basename(args.input_img))[0]
    
    save_image(srs, os.path.join(output_dir, f"{base_name}_inpainted.png"))
    save_image(gts, os.path.join(output_dir, f"{base_name}_gt.png"))
    save_image(lrs, os.path.join(output_dir, f"{base_name}_masked.png"))
    
    if gt_keep_masks is not None:
        save_image(gt_keep_masks, os.path.join(output_dir, f"{base_name}_mask.png"))
        print(f"- Mask visualization: {base_name}_mask.png")
    
    print(f"Results saved to {output_dir}")
    print(f"- Inpainted image: {base_name}_inpainted.png")
    print(f"- Ground truth: {base_name}_gt.png")
    print(f"- Masked input: {base_name}_masked.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single image inpainting with RePaint")
    parser.add_argument('--conf_path', type=str, required=True, 
                        help='Path to configuration file')
    parser.add_argument('--input-img', type=str, required=True,
                        help='Path to input image to be inpainted')
    parser.add_argument('--input-mask', type=str, required=True,
                        help='Path to mask image (black areas will be inpainted, white areas will be kept)')
    parser.add_argument('--input-gt', type=str, required=True,
                        help='Path to ground truth image')
    parser.add_argument('--output-dir', type=str, default='./output',
                        help='Directory to save output images (default: ./output)')
    
    args = parser.parse_args()
    
    # Load configuration
    conf_arg = conf_mgt.conf_base.Default_Conf()
    conf_arg.update(yamlread(args.conf_path))
    
    main(conf_arg, args)