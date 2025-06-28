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
from dataset import FilmDataset, TestDataset


device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_path, model):
    model = model().to(device)
    model.load_state_dict(torch.load(root_path + model_path))

    return model


def predict(model, img):


    return pred_mask

def test_model(model, model_name, images, device, new_loader=False):
    model.eval()
    model.to(device)
    model_name = model_name.split("/")[-1].split(".")[0]

    test_images = os.listdir(root_path + "/test")

    dataset = TestDataset(root_path+"/test", limit=None, new_loader=new_loader)

    

    start_time = time.time()
    for index, (image, img_path) in enumerate(dataset):
        #convert image tensor to numpy array
        image = image.permute(1, 2, 0).numpy()
        print(img_path)
        pred_mask = model(torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device))
        pred_mask = torch.sigmoid(pred_mask[0][0]).cpu().detach().numpy()
        # pred_mask = cv2.resize(pred_mask.detach().numpy(), (img.shape[1], img.shape[0]))
        folder_path = root_path + f"/results/{model_name}/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        filename = folder_path + f"{img_path.split('/')[-1].split('.')[0]}-mask_{model_name}.png"
        # print(filename)
        # im = Image.fromarray(pred_mask)
        # im.save(filename)
        if new_loader:
            pred_mask = torch.tensor(pred_mask)
            pred_mask = F.sigmoid(pred_mask,)
            pred_mask = pred_mask.detach().numpy()        
            pred_mask = pred_mask*255
            cv2.imwrite(filename, pred_mask)
        else:
            pred_mask = pred_mask*255
            cv2.imwrite(filename, pred_mask.astype(np.uint8))
        print(f"Saved {filename}")
        break
    print(f"Time taken: {time.time() - start_time}")

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Argument parser for U-Net inference")

    parser.add_argument("--input", type=str, help="Dataset to use", default="/test")
    parser.add_argument("--model", type=str, help="Path of model weights", default="/models/unet.pth")
    parser.add_argument("--name", type=str, help="Name of the model", default="model")
    parser.add_argument("--nou_loader", action="store_true", help="Use new loader", default=False)
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

    test_model(model, model_path, os.listdir(root_path + args.input), device, args.nou_loader)
    
    del model
    gc.collect()
    torch.cuda.empty_cache()