#!/usr/bin/env python3
"""
Batch Tiled Inpainting and Reconstruction Workflow.

This script automates the entire tiled inpainting process for a folder of images.
For each image, it:
1. Finds the corresponding mask in a specified mask folder.
2. Divides the image and its mask into 256x256 overlapping blocks.
3. Runs the RePaint model on each block to generate inpainted results.
4. Reconstructs the full 'inpainted', 'masked', and 'input' images from the tiles.
5. Cleans up the tile files automatically after each image is processed.

Optionally, it also processes a ground truth folder (--gt-folder) by dividing
and reconstructing those images without inpainting, allowing comparison.
"""

import os
import argparse
import shutil
from typing import List
import cv2
import numpy as np
from skimage.morphology import dilation, disk
# Import the necessary functions from our other scripts
# Ensure tiled_repaint.py and reconstruct.py are in the same directory
try:
    from tiled_repaint import (
        load_image_full_res, load_mask_full_res, divide_image_into_blocks,
        save_block_info, preprocess_block, toU8, save_image,
        create_model_and_diffusion, model_and_diffusion_defaults, select_args,
        classifier_defaults, create_classifier, NUM_CLASSES, dist_util, conf_mgt, yamlread
    )
    from reconstruct import (
        parse_block_info, reconstruct_image, save_reconstructed_image
    )
except ImportError as e:
    print(f"Error: Could not import from tiled_repaint.py or reconstruct.py.")
    print("Please ensure these files are in the same directory as this script.")
    print(f"Details: {e}")
    exit(1)

import torch as th
import torch.nn.functional as F
import time


def setup_model(conf: conf_mgt.Default_Conf):
    """Loads the RePaint model and diffusion setup once."""
    print("Setting up and loading RePaint model...")
    device = dist_util.dev(conf.get('device'))

    model, diffusion = create_model_and_diffusion(
        **select_args(conf, model_and_diffusion_defaults().keys()), conf=conf
    )
    model.load_state_dict(
        dist_util.load_state_dict(os.path.expanduser(conf.model_path), map_location="cpu")
    )
    model.to(device)
    if conf.use_fp16:
        model.convert_to_fp16()
    model.eval()

    cond_fn = None  # Classifier logic can be added here if needed

    def model_fn(x, t, y=None, gt=None, **kwargs):
        assert y is not None
        return model(x, t, y if conf.class_cond else None, gt=gt)
    
    print("Model loaded successfully.")
    return model, diffusion, model_fn, cond_fn, device


def generate_tiles_for_image(
    image_path: str,
    mask_array: List,
    temp_tile_dir: str,
    model_setup: tuple,
    conf: conf_mgt.Default_Conf,
    overlap: int,
):
    """Processes a single image: loads, tiles, inpaints, and saves all blocks."""
    model, diffusion, model_fn, cond_fn, device = model_setup

    print(f"Loading image: {os.path.basename(image_path)}")
    input_img = load_image_full_res(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Create dedicated output directories for this image's tiles
    dirs = ['blocks_input', 'blocks_mask', 'blocks_inpainted', 'blocks_masked']
    for d in dirs:
        os.makedirs(os.path.join(temp_tile_dir, d), exist_ok=True)
    
    print("Dividing into 256x256 blocks...")
    blocks = divide_image_into_blocks(input_img, mask_array, block_size=256, overlap=overlap)
    
    block_info_path = os.path.join(temp_tile_dir, f"{base_name}_block_info.txt")
    save_block_info(blocks, temp_tile_dir, base_name)
    print(f"Block information saved to temporary file: {block_info_path}")

    print(f"Inpainting {len(blocks)} blocks...")
    for block_idx, (img_block, mask_block, position) in enumerate(blocks):
        print(f"  - Processing block {block_idx + 1}/{len(blocks)} (row {position['row']}, col {position['col']})", end='\r')
        
        block_name_prefix = f"{base_name}_block_{block_idx:04d}_r{position['row']:02d}_c{position['col']:02d}"
        
        # Save input and mask blocks for reconstruction later
        save_image(img_block, os.path.join(temp_tile_dir, 'blocks_input', f"{block_name_prefix}_input.png"))
        save_image(mask_block, os.path.join(temp_tile_dir, 'blocks_mask', f"{block_name_prefix}_mask.png"))

        # Preprocess for model
        img_normalized, mask_normalized = preprocess_block(img_block, mask_block)
        img_tensor = th.from_numpy(img_normalized).unsqueeze(0).to(device)
        mask_tensor = th.from_numpy(mask_normalized).unsqueeze(0).unsqueeze(0).to(device)
        
        # Prepare model kwargs
        model_kwargs = {"gt": img_tensor.clone(), "gt_keep_mask": mask_tensor}
        if conf.class_cond:
             model_kwargs["y"] = th.ones(1, dtype=th.long, device=device) * conf.cond_y
        else:
             model_kwargs["y"] = th.randint(low=0, high=NUM_CLASSES, size=(1,), device=device)

        sample_fn = diffusion.p_sample_loop if not conf.use_ddim else diffusion.ddim_sample_loop
        
        # *** THE FIX IS HERE ***
        result = sample_fn(
            model_fn,
            (1, 3, 256, 256),
            clip_denoised=conf.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=device,
            progress=False,
            return_all=True,  # This crucial parameter was missing
            conf=conf
        )
        
        srs = toU8(result['sample'])
        lrs = toU8(result.get('gt') * model_kwargs.get('gt_keep_mask') + (-1) * th.ones_like(result.get('gt')) * (1 - model_kwargs.get('gt_keep_mask')))
        
        save_image(srs, os.path.join(temp_tile_dir, 'blocks_inpainted', f"{block_name_prefix}_inpainted.png"))
        save_image(lrs, os.path.join(temp_tile_dir, 'blocks_masked', f"{block_name_prefix}_masked.png"))

    print(f"\nFinished inpainting all {len(blocks)} blocks for {os.path.basename(image_path)}.")
    return base_name, block_info_path


def divide_and_reconstruct_gt(
    gt_image_path: str,
    temp_tile_dir: str,
    conf: conf_mgt.Default_Conf,
    overlap: int
):
    """
    Divide GT image into blocks and reconstruct it,
    saving tiles and reconstructed image for comparison.
    """
    print(f"\nProcessing GT image: {os.path.basename(gt_image_path)}")
    input_img = load_image_full_res(gt_image_path)
    base_name = os.path.splitext(os.path.basename(gt_image_path))[0]

    # Create dedicated output directories for GT blocks
    dirs = ['blocks_gt']
    for d in dirs:
        os.makedirs(os.path.join(temp_tile_dir, d), exist_ok=True)

    print("Dividing GT image into 256x256 blocks...")
    # Divide image into blocks with no mask, just the image blocks
    blocks = divide_image_into_blocks(input_img, mask_array=None, block_size=256, overlap=overlap)

    # Save blocks and block info
    save_block_info(blocks, temp_tile_dir, base_name)

    for block_idx, (img_block, _, position) in enumerate(blocks):
        block_name_prefix = f"{base_name}_block_{block_idx:04d}_r{position['row']:02d}_c{position['col']:02d}"
        save_image(img_block, os.path.join(temp_tile_dir, 'blocks_gt', f"{block_name_prefix}_gt.png"))

    print("Reconstructing GT image from blocks...")
    blocks_info = parse_block_info(os.path.join(temp_tile_dir, f"{base_name}_block_info.txt"))

    reconstructed_img = reconstruct_image(
        blocks_info=blocks_info,
        block_dir=temp_tile_dir,
        base_name=base_name,
        suffix='gt',
        overlap=overlap,
        blend_overlaps=True
    )

    return base_name, reconstructed_img


def main():
    parser = argparse.ArgumentParser(description="Batch Tiled Inpainting and Reconstruction Workflow")
    parser.add_argument('--input-folder', type=str, required=True, help='Folder containing input images to be inpainted')
    parser.add_argument('--mask-folder', type=str, required=True, help='Folder containing masks. Each mask must have the same name as its corresponding image.')
    parser.add_argument('--output-folder', type=str, required=True, help='Folder to save the final reconstructed images')
    parser.add_argument('--conf_path', type=str, required=True, help='Path to the RePaint model configuration file')
    parser.add_argument('--overlap', type=int, default=32, help='Overlap between blocks in pixels (default: 32)')
    parser.add_argument('--gt-folder', type=str, default=None, help='Optional folder containing ground truth images to divide and reconstruct')

    args = parser.parse_args()

    # --- Setup ---
    os.makedirs(args.output_folder, exist_ok=True)

    # Load RePaint configuration and model
    conf_arg = conf_mgt.conf_base.Default_Conf()
    conf_arg.update(yamlread(args.conf_path))
    model_setup = setup_model(conf_arg)

    # Find images to process
    image_files = sorted([f for f in os.listdir(args.input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not image_files:
        print(f"No images found in {args.input_folder}. Exiting.")
        return

    print(f"Found {len(image_files)} images to process.")

    # --- Main Loop ---
    for image_file in image_files:
        image_path = os.path.join(args.input_folder, image_file)
        base_name = os.path.splitext(image_file)[0]
        
        # Find corresponding mask file
        mask_path = os.path.join(args.mask_folder, image_file)
        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for {image_file} at {mask_path}. Skipping this image.")
            continue

        temp_tile_dir = os.path.join(args.output_folder, f"temp_tiles_{base_name}")

        print("\n" + "="*50)
        print(f"PROCESSING IMAGE: {image_file}")
        print("="*50)

        try:
            # Load the specific mask for this image
            print(f"Loading mask: {os.path.basename(mask_path)}")
            mask_array = load_mask_full_res(mask_path)
            # Dilate the mask by 3 pixels
            selem = disk(10)  # Structuring element of radius ~10
            mask_array = dilation(mask_array, selem).astype(np.uint8)  # Ski
            # Step 1: Generate all tiles for the current image
            _, block_info_path = generate_tiles_for_image(
                image_path=image_path,
                mask_array=mask_array,
                temp_tile_dir=temp_tile_dir,
                model_setup=model_setup,
                conf=conf_arg,
                overlap=args.overlap
            )

            # Step 2: Reconstruct all required versions
            print("\nStarting reconstruction...")
            blocks_info = parse_block_info(block_info_path)
            
            reconstruction_types = ['inpainted', 'masked', 'input']
            for recon_type in reconstruction_types:
                print(f"  - Reconstructing '{recon_type}' version...")
                
                reconstructed_img = reconstruct_image(
                    blocks_info=blocks_info,
                    block_dir=temp_tile_dir,
                    base_name=base_name,
                    suffix=recon_type,
                    overlap=args.overlap,
                    blend_overlaps=True
                )
                
                output_path = os.path.join(args.output_folder, f"{base_name}_reconstructed_{recon_type}.png")
                save_reconstructed_image(reconstructed_img, output_path)

        except Exception as e:
            print(f"\n\nERROR processing {image_file}: {e}")
            import traceback
            traceback.print_exc()
            print("Skipping to next image.")
        finally:
            # Step 3: Clean up tiles for the current image
            if os.path.exists(temp_tile_dir):
                print(f"\nCleaning up temporary tile directory: {temp_tile_dir}")
                shutil.rmtree(temp_tile_dir)

    # --- Optional GT Folder Processing ---
    if args.gt_folder is not None:
        gt_files = sorted([f for f in os.listdir(args.gt_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not gt_files:
            print(f"No GT images found in {args.gt_folder}. Skipping GT processing.")
        else:
            print(f"\nFound {len(gt_files)} GT images to process.")
            for gt_file in gt_files:
                gt_path = os.path.join(args.gt_folder, gt_file)
                base_name = os.path.splitext(gt_file)[0]
                temp_tile_dir = os.path.join(args.output_folder, f"temp_tiles_gt_{base_name}")
                try:
                    _, reconstructed_gt = divide_and_reconstruct_gt(
                        gt_image_path=gt_path,
                        temp_tile_dir=temp_tile_dir,
                        conf=conf_arg,
                        overlap=args.overlap
                    )

                    output_path = os.path.join(args.output_folder, f"{base_name}_reconstructed_gt.png")
                    save_reconstructed_image(reconstructed_gt, output_path)

                except Exception as e:
                    print(f"\nERROR processing GT image {gt_file}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                finally:
                    if os.path.exists(temp_tile_dir):
                        shutil.rmtree(temp_tile_dir)

    print("\nBatch processing complete.")

if __name__ == "__main__":
    main()