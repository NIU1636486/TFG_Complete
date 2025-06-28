from damage_generator.scans import load_scans, load_all_synthetic_images
from damage_generator.generate_masks import create_random_mask, create_random_mask_with_transparency
import os
import argparse
import cv2 as cv
import uuid
import numpy as np
from matplotlib import pyplot as plt
import warnings
import random 
factorize = True
n = 3000
warnings.simplefilter(action='ignore', category=FutureWarning)
height = 1024
width = 1024

synthetic = False
rescale = True
binarized = True
verbose = False
uniform = False

scans_path = "/scans/"
synthetic_path = "/synthetic/"

abs_path = os.path.abspath("")
scans_path = abs_path + scans_path
synthetic_path = os.path.dirname(os.path.normpath(abs_path))+ synthetic_path
df_artifacts = load_scans(scans_path, verbose=verbose)


### Optionally use synthetic artifacts ###
if synthetic:
    df_synthetic = df_synthetic_articats = load_all_synthetic_images(synthetic_path, verbose=verbose)
else: df_synthetic = None

df_per_patch_counts = df_artifacts.groupby(['Quandrant', 'Type']).size().to_frame('Counts').reset_index()

if factorize:
    factor = random.randint(1, 20)
    df_per_patch_counts_rescaled = df_per_patch_counts.copy()
    df_per_patch_counts_rescaled['Counts'] = df_per_patch_counts_rescaled['Counts'] * factor



def generate_n_masks(height, width, n, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for i in range(n):
        print(f"Generating mask {i}")

        mask, binary_mask, perlin_noise = create_random_mask_with_transparency((height, width), df_artifacts, df_synthetic, df_per_patch_counts_rescaled, use_synthetic=synthetic, rescale=rescale, uniform_sample=uniform, verbose=verbose, alpha_prob = 0.7)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        print(f"Saving mask {i}")
        cv.imwrite(output_folder+"mask_"+str(i)+".png", mask)

mask = generate_n_masks(752, 1920, n, "./masks_random_intensity/")
import requests
requests.post("https://ntfy.sh/tfg-pol", data = "Finished generating masks",
            headers={"title": "Mask generation",})