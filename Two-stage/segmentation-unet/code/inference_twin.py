from network import U_Net, R2AttU_Net, AttU_Net,R2U_Net_Reduced, R2AttU_Net_Reduced
import copy
import os
import random
import shutil
import zipfile
from math import atan2, cos, sin, sqrt, pi, log

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from numpy import linalg as LA
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from tqdm import tqdm
import gc
import time

import argparse
from dataset import FilmDataset


device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_path, model):
    model = model().to(device)
    model.load_state_dict(torch.load(root_path + model_path))

    return model


def predict(model, img):
    pred_mask = model(img.to(device))
    pred_mask = torch.sigmoid(pred_mask[0][0]).cpu().detach().numpy()

    return pred_mask

def test_model(model, model_name, images, device):
    model.eval()
    model.to(device)
    model_name = model_name.split("/")[-1].split(".")[0]

    test_images = os.listdir(root_path + "/twin_peaks/input")
    print(test_images)

    



    start_time = time.time()
    for index, image in enumerate(test_images):
        #load image
        img = cv2.imread(root_path + "/twin_peaks/input/" + image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # image = img

        # resize to 1024x1024
        img = cv2.resize(img, (1024, 1024))
        img = img / 255.0

        # convert image to tensor
        img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
        pred_mask = predict(model, img)
        # pred_mask = cv2.resize(pred_mask.detach().numpy(), (img.shape[1], img.shape[0]))
        print(root_path)
        folder_path = root_path + f"/results_twin_peaks/{model_name}/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        filename = folder_path + f"{image}_mask_{model_name}_{index}.png"
        # print(filename)
        # im = Image.fromarray(pred_mask)
        # im.save(filename)
        cv2.imwrite(filename, pred_mask*255)
    print(f"Time taken: {time.time() - start_time}")

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Argument parser for U-Net inference")

    parser.add_argument("--input", type=str, help="Dataset to use", default="/test")
    parser.add_argument("--model", type=str, help="Path of model weights", default="/models/unet.pth")
    parser.add_argument("--name", type=str, help="Name of the model", default="model")
    args=parser.parse_args()

    root_path = "/hhome/priubrogent/tfg/segmenter-unet"

    model_path = args.model
    print(f"Testing model {model_path}")
    model_name = model_path.split("/")[-1].split(".")[0]

    match args.name:
        case "unet":
            model = U_Net()
        case "r2attunet":
            model = R2AttU_Net()
        case "attunet":
            model = AttU_Net()
        case "r2attunetreduced":
            model = R2AttU_Net_Reduced()

    model = model.to(device)
    model.load_state_dict(torch.load(root_path + model_path))

    test_model(model, model_path, os.listdir(root_path + args.input), device)
    
    del model
    gc.collect()
    torch.cuda.empty_cache()