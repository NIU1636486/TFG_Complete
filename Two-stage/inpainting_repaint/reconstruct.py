#!/usr/bin/env python3
"""
Image reconstruction from tiled blocks.
Reconstructs full images from blocks processed by the tiled inpainting script.
Handles overlapping regions with smooth blending.
"""

import os
import argparse
import numpy as np
from PIL import Image
import re
from typing import List, Tuple, Dict, Optional

def parse_block_info(info_file: str) -> List[Dict]:
    """
    Parse block information from the block info file.
    
    Args:
        info_file: Path to the block info file
        
    Returns:
        List of dictionaries containing block information
    """
    blocks_info = []
    
    with open(info_file, 'r') as f:
        lines = f.readlines()
    
    data_lines = [line.strip() for line in lines if line.strip() and not line.startswith('Total') 
                  and not line.startswith('Block information') and not line.startswith('Format')]
    
    for line in data_lines:
        if not line:
            continue
            
        parts = [part.strip() for part in line.split(',')]
        if len(parts) >= 9:
            block_info = {
                'block_id': int(parts[0]),
                'row': int(parts[1]),
                'col': int(parts[2]),
                'start_x': int(parts[3]),
                'start_y': int(parts[4]),
                'end_x': int(parts[5]),
                'end_y': int(parts[6]),
                'original_w': int(parts[7]),
                'original_h': int(parts[8])
            }
            blocks_info.append(block_info)
    
    return blocks_info

def load_block_image(block_dir: str, base_name: str, block_info: Dict, suffix: str = "inpainted") -> np.ndarray:
    """
    Load a specific block image. With the new tiling, it's not cropped here.
    """
    block_name = f"{base_name}_block_{block_info['block_id']:04d}_r{block_info['row']:02d}_c{block_info['col']:02d}_{suffix}.png"
    
    # Fix: Construct the correct path with the subdirectory
    subdirectory = f"blocks_{suffix}"
    block_path = os.path.join(block_dir, subdirectory, block_name)
    
    if not os.path.exists(block_path):
        raise FileNotFoundError(f"Block image not found: {block_path}")
    
    block_img = np.array(Image.open(block_path))
    
    # With the new tiling logic, original_h/w will be the full block size, so
    # this cropping step becomes a no-op, which is fine.
    # It ensures backward compatibility if you have old, padded blocks.
    original_h = block_info['original_h']
    original_w = block_info['original_w']
    
    actual_h, actual_w = block_img.shape[:2]
    crop_h = min(original_h, actual_h)
    crop_w = min(original_w, actual_w)
    
    return block_img[:crop_h, :crop_w]

def calculate_full_image_size(blocks_info: List[Dict]) -> Tuple[int, int]:
    """
    Calculate the size of the full reconstructed image from the logical grid ends.
    """
    max_y = max(block['end_y'] for block in blocks_info)
    max_x = max(block['end_x'] for block in blocks_info)
    
    return max_y, max_x

def determine_overlap_position(block_info: Dict, all_blocks: List[Dict]) -> List[str]:
    """
    Determine which edges of a block have overlaps with neighboring blocks.
    """
    positions = []
    row, col = block_info['row'], block_info['col']
    
    has_top = any(b['row'] == row - 1 and b['col'] == col for b in all_blocks)
    has_bottom = any(b['row'] == row + 1 and b['col'] == col for b in all_blocks)
    has_left = any(b['row'] == row and b['col'] == col - 1 for b in all_blocks)
    has_right = any(b['row'] == row and b['col'] == col + 1 for b in all_blocks)
    
    if has_top: positions.append('top')
    if has_bottom: positions.append('bottom')
    if has_left: positions.append('left')
    if has_right: positions.append('right')
    
    return positions

def reconstruct_image(blocks_info: List[Dict], block_dir: str, base_name: str, 
                     suffix: str = "inpainted", overlap: int = 32, 
                     blend_overlaps: bool = True) -> np.ndarray:
    """
    Reconstruct the full image from blocks.
    """
    full_h, full_w = calculate_full_image_size(blocks_info)
    first_block = load_block_image(block_dir, base_name, blocks_info[0], suffix)
    channels = 3 if len(first_block.shape) == 3 and first_block.shape[2] == 3 else 1
    
    dtype = np.float64
    if channels == 3:
        full_image = np.zeros((full_h, full_w, 3), dtype=dtype)
        weight_map = np.zeros((full_h, full_w), dtype=dtype)
    else:
        full_image = np.zeros((full_h, full_w), dtype=dtype)
        weight_map = np.zeros((full_h, full_w), dtype=dtype)
    
    print(f"Reconstructing image of size {full_w}x{full_h} from {len(blocks_info)} blocks")
    print(f"Using overlap blending with {overlap} pixel overlap" if blend_overlaps and overlap > 0 else "No overlap blending")

    for i, block_info in enumerate(blocks_info):
        print(f"Processing block {i+1}/{len(blocks_info)} (row {block_info['row']}, col {block_info['col']})")
        
        block_img = load_block_image(block_dir, base_name, block_info, suffix).astype(dtype)
        
        start_y, start_x = block_info['start_y'], block_info['start_x']
        actual_h, actual_w = block_img.shape[:2]
        end_y, end_x = start_y + actual_h, start_x + actual_w
        
        weights = np.ones((actual_h, actual_w), dtype=dtype)
        if blend_overlaps and overlap > 0:
            overlap_positions = determine_overlap_position(block_info, blocks_info)
            if overlap_positions:
                fade_size = min(overlap, min(actual_h, actual_w) // 2)
                
                # Consistent fade-in ramp (0 to 1) for all edges
                fade = np.linspace(0, 1, fade_size, dtype=dtype)
                
                if 'top' in overlap_positions:
                    weights[:fade_size, :] *= fade[:, np.newaxis]
                if 'left' in overlap_positions:
                    weights[:, :fade_size] *= fade[np.newaxis, :]
                if 'bottom' in overlap_positions:
                    weights[actual_h-fade_size:, :] *= np.flip(fade)[:, np.newaxis]
                if 'right' in overlap_positions:
                    weights[:, actual_w-fade_size:] *= np.flip(fade)[np.newaxis, :]

        if channels == 3:
            full_image[start_y:end_y, start_x:end_x, :] += block_img * weights[:, :, np.newaxis]
        else:
            full_image[start_y:end_y, start_x:end_x] += block_img * weights
        
        weight_map[start_y:end_y, start_x:end_x] += weights
    
    # Normalize by weights
    weight_map_mask = weight_map > 1e-6
    if channels == 3:
        full_image[weight_map_mask] /= weight_map[weight_map_mask][:, np.newaxis]
    else:
        full_image[weight_map_mask] /= weight_map[weight_map_mask]

    if np.any(~weight_map_mask):
        print(f"Warning: Found {np.sum(~weight_map_mask)} pixels with zero weight")

    return np.clip(full_image, 0, 255).astype(np.uint8)

def save_reconstructed_image(image: np.ndarray, output_path: str):
    """Save reconstructed image."""
    Image.fromarray(image).save(output_path)
    print(f"Reconstructed image saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Reconstruct full image from tiled blocks")
    parser.add_argument('--blocks-dir', type=str, required=True, help='Directory containing the block images')
    parser.add_argument('--block-info', type=str, required=True, help='Path to block info file')
    parser.add_argument('--output', type=str, required=True, help='Output path for reconstructed image')
    parser.add_argument('--block-type', type=str, default='inpainted', choices=['inpainted', 'masked', 'input'], help='Type of blocks to reconstruct (default: inpainted)')
    parser.add_argument('--overlap', type=int, default=32, help='Overlap size used during tiling (default: 32)')
    parser.add_argument('--no-blend', action='store_true', help='Disable blending of overlapping regions')
    parser.add_argument('--base-name', type=str, default=None, help='Base name for block files (auto-detected if not provided)')

    args = parser.parse_args()
    
    print("Reading block information...")
    blocks_info = parse_block_info(args.block_info)
    print(f"Found {len(blocks_info)} blocks")
    
    if args.base_name is None:
        args.base_name = os.path.basename(args.block_info).replace('_block_info.txt', '')
    
    # Fix: Remove the block_dir_map since we'll handle subdirectories in load_block_image
    if not os.path.exists(args.blocks_dir):
        raise FileNotFoundError(f"Block directory not found: {args.blocks_dir}")

    print(f"Reconstructing from {args.block_type} blocks in {args.blocks_dir}")
    
    reconstructed = reconstruct_image(
        blocks_info=blocks_info,
        block_dir=args.blocks_dir,
        base_name=args.base_name,
        suffix=args.block_type,
        overlap=args.overlap,
        blend_overlaps=not args.no_blend
    )
    
    save_reconstructed_image(reconstructed, args.output)

if __name__ == "__main__":
    main()