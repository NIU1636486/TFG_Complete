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


from unet_model import UNet

import argparse
parser=argparse.ArgumentParser(description="Argument parser for U-Net")

parser.add_argument("--wandb", action="store_true", help="Use wandb for logging", default=False)
parser.add_argument("--noti", action="store_true", help="Send phone notifications", default=False)
parser.add_argument("--epochs", type=int, help="Number of epochs", default=20)
parser.add_argument("--dataset", type=str, help="Dataset to use", default="/leone")
parser.add_argument("--output-name", type=str, help="Output name for the model", default="model.pth")
parser.add_argument("--run-name", type=str, help="Run name for wandb", default="U-Net-Training")
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

class FilmDataset(Dataset):
    def __init__(self, root_path, limit=None):
        self.root_path = root_path
        self.limit = limit
        self.images = sorted([os.path.join(root_path, "train", i) for i in os.listdir(os.path.join(root_path, "train"))])[:self.limit]
        self.masks = sorted([os.path.join(root_path, "train_mask", i) for i in os.listdir(os.path.join(root_path, "train_mask"))])[:self.limit]

        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor()])

        if self.limit is None:
            self.limit = len(self.images)

    def __getitem__(self, index):
        # Read the image and mask using skimage
        img = imread(self.images[index])  # This will load the image as a numpy array
        mask = imread(self.masks[index], as_gray=True)  # Load mask in grayscale (0 or 1)

        # Resize the images and mask
        img_resized = resize(img, (1024, 1024), mode='reflect', anti_aliasing=True)
        mask_resized = resize(mask, (1024, 1024), mode='reflect', anti_aliasing=True)

        # Binarize the mask: 0 for 255, 1 for everything else
        mask_resized = np.where(mask_resized == 1, 0, 1)  # Assumes the mask is either 0 or 1 after resizing

        # Convert numpy arrays to PyTorch tensors
        img_tensor = torch.tensor(img_resized).permute(2, 0, 1).float()  # Convert RGB image to tensor and rearrange channels
        mask_tensor = torch.tensor(mask_resized).unsqueeze(0).float()  # Convert mask to tensor, adding a channel dimension

        # Normalize the image if needed
        return img_tensor, mask_tensor

    def __len__(self):
        return min(len(self.images), self.limit)




def dice_coefficient(prediction, target, epsilon=1e-07):
    prediction_copy = prediction.clone()

    prediction_copy[prediction_copy < 0] = 0
    prediction_copy[prediction_copy > 0] = 1

    intersection = abs(torch.sum(prediction_copy * target))
    union = abs(torch.sum(prediction_copy) + torch.sum(target))
    dice = (2. * intersection + epsilon) / (union + epsilon)
    
    return dice

####### HYPERPARAMETERS ########
LEARNING_RATE = 3e-4
BATCH_SIZE = 8
EPOCHS = args.epochs

############ DATA PREPARATION ############

print("Creant dataset...")
train_dataset = FilmDataset(dataset_root, limit=None)
generator = torch.Generator().manual_seed(25)

train_dataset, test_dataset = random_split(train_dataset, [0.8, 0.2], generator=generator)
test_dataset, val_dataset = random_split(test_dataset, [0.5, 0.5], generator=generator)

train_dataloader = DataLoader(dataset=train_dataset,
                              num_workers=num_workers, pin_memory=False,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset,
                            num_workers=num_workers, pin_memory=False,
                            batch_size=BATCH_SIZE,
                            shuffle=True)

test_dataloader = DataLoader(dataset=test_dataset,
                            num_workers=num_workers, pin_memory=False,
                            batch_size=BATCH_SIZE,
                            shuffle=True)

model = UNet(in_channels=3, num_classes=1).to(device)



optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()


if NOTIFICATIONS:
    send_notification("Starting training 🚀")


train_losses = []
train_dcs = []
val_losses = []
val_dcs = []

if WANDB:
    wandb.init(
        project="tfg-segmentador",
        name=args.run_name, 
        config={
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "architecture": "U-Net",
        "dataset": args.dataset,
        }
    )

######### TRAINING #########

for epoch in tqdm(range(EPOCHS)):
    model.train()
    train_running_loss = 0
    train_running_dc = 0
    
    for idx, img_mask in enumerate(tqdm(train_dataloader, position=0, leave=True)):
        img = img_mask[0].float().to(device)
        mask = img_mask[1].float().to(device)
        
        y_pred = model(img)
        optimizer.zero_grad()
        
        dc = dice_coefficient(y_pred, mask)
        loss = criterion(y_pred, mask)
        
        train_running_loss += loss.item()
        train_running_dc += dc.item()

        loss.backward()
        optimizer.step()

    train_loss = train_running_loss / (idx + 1)
    train_dc = train_running_dc / (idx + 1)
    
    train_losses.append(train_loss)
    train_dcs.append(train_dc)

    model.eval()
    val_running_loss = 0
    val_running_dc = 0
    
    with torch.no_grad():
        for idx, img_mask in enumerate(tqdm(val_dataloader, position=0, leave=True)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)

            y_pred = model(img)
            loss = criterion(y_pred, mask)
            dc = dice_coefficient(y_pred, mask)
            
            val_running_loss += loss.item()
            val_running_dc += dc.item()

        val_loss = val_running_loss / (idx + 1)
        val_dc = val_running_dc / (idx + 1)
    
    val_losses.append(val_loss)
    val_dcs.append(val_dc)

    print("-" * 30)
    print(f"Training Loss EPOCH {epoch + 1}: {train_loss:.4f}")
    print(f"Training DICE EPOCH {epoch + 1}: {train_dc:.4f}")
    if WANDB:
        wandb.log({"epoch": epoch, "train_loss": train_loss, "train_dc": train_dc, "val_loss": val_loss, "val_dc": val_dc, "images": [
        wandb.Image(img[0].cpu().detach().numpy().transpose(1, 2, 0)),
        wandb.Image(mask[0].cpu().detach().numpy().transpose(1, 2, 0)),
        wandb.Image(y_pred[0].cpu().detach().numpy().transpose(1, 2, 0))
    ]})
        
    print("\n")
    print(f"Validation Loss EPOCH {epoch + 1}: {val_loss:.4f}")
    print(f"Validation DICE EPOCH {epoch + 1}: {val_dc:.4f}")
    print("-" * 30)

# Saving the model
model = model.to("cpu")
torch.save(model.state_dict(), args.output_name)

if NOTIFICATIONS:
    send_notification("Training finished 😀")
