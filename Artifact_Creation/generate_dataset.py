import os
import cv2 as cv
import numpy as np
import pandas as pd
from damage_generator.scans import load_scans, load_all_synthetic_images

def add_artifact_masks(test_images_dir, output_dir, scans_dir, synthetic_dir, use_synthetic=False, verbose=False):
    # Load real artifact masks
    df_artifacts = load_scans(scans_dir, verbose=verbose)
    
    # Optionally load synthetic artifact masks
    df_synthetic = None
    if use_synthetic:
        df_synthetic = load_all_synthetic_images(synthetic_dir, verbose=verbose)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each test image
    for img_filename in os.listdir(test_images_dir):
        if img_filename.lower().endswith(('.jpg', '.png')):
            img_path = os.path.join(test_images_dir, img_filename)
            img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
            
            # Apply real artifact masks
            for _, artifact in df_artifacts.iterrows():
                apply_mask(img, artifact)
            
            # Apply synthetic artifact masks if available
            if df_synthetic is not None:
                for _, artifact in df_synthetic.iterrows():
                    apply_mask(img, artifact)
            
            # Save the modified image
            output_path = os.path.join(output_dir, img_filename)
            cv.imwrite(output_path, img)
            if verbose:
                print(f"Processed and saved {output_path}")

def apply_mask(img, artifact):
    # Extract mask details
    x, y, w, h = artifact['bbox x'], artifact['bbox y'], artifact['bbox w'], artifact['bbox h']
    mask = artifact['Artifact']
    
    # Ensure mask is resized to match the target region
    mask = cv.resize(mask, (w, h), interpolation=cv.INTER_NEAREST)

    # Convert to grayscale if needed and ensure it is 8-bit
    if len(mask.shape) == 3:  # If mask is RGB, convert to grayscale
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    mask = mask.astype(np.uint8)  # Convert to correct type

    # Ensure the image region has the same size
    img_region = img[y:y+h, x:x+w]

    if img_region.shape[:2] != mask.shape:
        print(f"Size mismatch! Image region: {img_region.shape}, Mask: {mask.shape}")
        return  # Skip this mask to avoid errors

    # Apply mask using OpenCV copyTo (more reliable than bitwise operations)
    cv.copyTo(img_region, mask, dst=img_region)


# Example usage
test_images_dir = 'test_images/'
output_dir = 'output_images/'
scans_dir = 'scans/'
synthetic_dir = 'synthetic/'
add_artifact_masks(test_images_dir, output_dir, scans_dir, synthetic_dir, use_synthetic=False, verbose=False)
