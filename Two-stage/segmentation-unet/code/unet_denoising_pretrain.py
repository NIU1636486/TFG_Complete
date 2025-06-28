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
from skimage.io import imread
from skimage.transform import resize


from network import U_Net, R2AttU_Net, AttU_Net, R2AttU_Net_Reduced
from evaluation import *
from dataset import DenoisingDataset

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


import argparse
parser=argparse.ArgumentParser(description="Argument parser for U-Net")

parser.add_argument("--wandb", action="store_true", help="Use wandb for logging", default=False)
parser.add_argument("--noti", action="store_true", help="Send phone notifications", default=False)
parser.add_argument("--epochs", type=int, help="Number of epochs", default=20)
parser.add_argument("--dataset", type=str, help="Dataset to use", default="/leone")
parser.add_argument("--output-name", type=str, help="Output name for the model", default="model")
parser.add_argument("--run-name", type=str, help="Run name for wandb", default="U-Net-Training")
parser.add_argument("--model-type", type=str, help="Model type", default="unet")
parser.add_argument("--batch-size", type=int, help="Batch size", default=2)
args=parser.parse_args()

WANDB = args.wandb
NOTIFICATIONS = args.noti
if NOTIFICATIONS:
    from notifications import send_notification

dataset_root = "/hhome/priubrogent/tfg/segmenter-unet" + args.dataset

if WANDB:
    import wandb
    wandb.login(key="8e9b2ed0a8b812e7888f16b3aa28491ba440d81a")



device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    num_workers = torch.cuda.device_count() * 4
else:
    num_workers = 1

num_workers = 4

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage.io import imread
from skimage.transform import resize
import numpy as np



BATCH_SIZE = args.batch_size
lr = 3e-4
num_epochs = args.epochs
num_epochs_decay = 10

print("Creant dataset...")

denoise_dataset = DenoisingDataset(root_path=dataset_root, height=1024, width=1024, limit=None, seed=42)
generator = torch.Generator().manual_seed(25)

train_dataset, test_dataset = random_split(denoise_dataset, [0.8, 0.2], generator=generator)
test_dataset, val_dataset = random_split(test_dataset, [0.5, 0.5], generator=generator)

train_loader = DataLoader(dataset=train_dataset,
                              num_workers=num_workers, pin_memory=False,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
valid_loader = DataLoader(dataset=val_dataset,
                            num_workers=num_workers, pin_memory=False,
                            batch_size=BATCH_SIZE,
                            shuffle=True)


match args.model_type:
    case "unet":
        unet = U_Net(output_ch=3).to(device)
    case "r2attunet":
        unet = R2AttU_Net(output_ch=3).to(device)
    case "attunet":
        unet = AttU_Net(output_ch=3).to(device)
    case "r2attunetreduced":
        unet = R2AttU_Net_Reduced(output_ch=3).to(device)


criterion = torch.nn.MSELoss()

optimizer = torch.optim.Adam(unet.parameters(), lr=3e-4) 

num_epochs_decay = args.epochs // 2 
model_type = args.run_name # Set model type
unet_path = "pretrained_"+args.output_name+"_"+args.model_type+"_"+str(num_epochs)+"epochs"+"_"+str(len(denoise_dataset))+"images"+".pth"  # Set model path 

def reset_grad():
    optimizer.zero_grad()


if WANDB:
    wandb.init(
        project="tfg-pretraining",
        name=args.run_name + "_" + args.model_type + "_" + str(num_epochs) + "epochs", 
        config={
        "learning_rate": lr,
        "batch_size":BATCH_SIZE,
        "epochs": num_epochs,
        "architecture": args.model_type,
        "dataset": args.dataset,
        "dataset_size": len(denoise_dataset),
        }
    )

if NOTIFICATIONS:
    send_notification(f"Starting training of denoising {args.model_type} 🚀")

for epoch in range(num_epochs):

    unet.train()
    epoch_loss = 0
    psnr_acc = 0
    length = 0
    val_loss_total = 0

    for images, GT in train_loader:
        images, GT = images.to(device), GT.to(device)



        SR = unet(images)
        loss = criterion(SR, GT)
        epoch_loss += loss.item() * images.size(0)

        # Backpropagation
        reset_grad()
        loss.backward()
        optimizer.step()

        # Compute metrics
        psnr_acc += psnr(SR, GT).item()
        length += images.size(0)

    avg_psnr = psnr_acc / length
    epoch_loss /= length


    # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, \n'
    #       f'[Training] psnr: {psnr:.4f}')
    
    if WANDB:
        wandb.log({"epoch": epoch, "train_loss": epoch_loss, "train_psnr": avg_psnr,
        "images": [
        wandb.Image(images[0].cpu().detach().numpy().transpose(1, 2, 0)),
        wandb.Image(GT[0].cpu().detach().numpy().transpose(1, 2, 0)),
        wandb.Image(SR[0].cpu().detach().numpy().transpose(1, 2, 0))
    ]})

    # Decay learning rate
    if (epoch + 1) > (num_epochs - num_epochs_decay):
        lr -= (lr / float(num_epochs_decay))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f'Decay learning rate to lr: {lr}.')

    # Validation
    unet.eval()

    val_psnr_acc = 0
    length = 0


    with torch.no_grad():
        for noisy_imgs, clean_imgs in valid_loader:
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
            SR = unet(noisy_imgs)

            val_loss = criterion(SR, clean_imgs)
            val_loss_total += val_loss.item() * noisy_imgs.size(0)
            val_psnr_acc += psnr(SR, clean_imgs).item()
            length += noisy_imgs.size(0)

    avg_val_psnr = val_psnr_acc / length
    # print(f'[Validation] Avg PSNR: {avg_val_psnr:.2f} dB')
    val_loss = val_loss_total / length
    if WANDB:
        wandb.log({
            "val_psnr": avg_val_psnr,
            "val_loss": val_loss,
            "epoch": epoch,
            "images_val": [
                wandb.Image(noisy_imgs[0].cpu().numpy().transpose(1, 2, 0)),
                wandb.Image(clean_imgs[0].cpu().numpy().transpose(1, 2, 0)),
                wandb.Image(SR[0].cpu().detach().numpy().transpose(1, 2, 0))
            ]
        })

model = unet.to("cpu")
torch.save(model.state_dict(), unet_path)

if NOTIFICATIONS:
    send_notification(f"Training finished of denoising {args.model_type}  🎉")

if WANDB:
    wandb.finish()

import gc
gc.collect()
torch.cuda.empty_cache()
