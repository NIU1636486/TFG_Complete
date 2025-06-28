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
from dataset import FilmDataset

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
parser.add_argument("--pretrained", type=str, help="Pretrained model path", default=None)
parser.add_argument("--nou_loader", action="store_true", help="Use wandb for logging", default=False)
parser.add_argument("--debug", action="store_true", help="Debug mode", default=False)

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

if args.debug:
    device = "cpu"

if device == "cuda":
    num_workers = torch.cuda.device_count() * 4
else:
    num_workers = 1

num_workers = 1

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
film_dataset = FilmDataset(dataset_root, limit=None, seed=42, export_csv=True, csv_path=args.output_name+"_"+args.model_type+"_"+str(num_epochs)+"epochs_dataset_map.csv", new_method=args.nou_loader, sigma=1)

generator = torch.Generator().manual_seed(25)

train_dataset, test_dataset = random_split(film_dataset, [0.8, 0.2], generator=generator)
test_dataset, val_dataset = random_split(test_dataset, [0.5, 0.5], generator=generator)

train_loader = DataLoader(dataset=train_dataset,
                              num_workers=num_workers, pin_memory=False,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
valid_loader = DataLoader(dataset=val_dataset,
                            num_workers=num_workers, pin_memory=False,
                            batch_size=BATCH_SIZE,
                            shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                            num_workers=num_workers, pin_memory=False,
                            batch_size=BATCH_SIZE,
                            shuffle=True)



match args.model_type:
    case "unet":
        if args.pretrained:
            unet = U_Net(output_ch=3)
        else:
            unet = U_Net().to(device)
    case "r2attunet":
        unet = R2AttU_Net().to(device)
    case "attunet":
        unet = AttU_Net().to(device)
    case "r2attunetreduced":
        unet = R2AttU_Net_Reduced().to(device)

if args.pretrained:
    unet.load_state_dict(torch.load(args.pretrained, map_location=device), strict=False)
    unet.Conv_1x1 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)

    nn.init.xavier_uniform_(unet.Conv_1x1.weight)
    nn.init.zeros_(unet.Conv_1x1.bias)
    unet = unet.to(device)

criterion = torch.nn.BCELoss() 
optimizer = torch.optim.Adam(unet.parameters(), lr=3e-4) 
num_epochs_decay = args.epochs // 2 
model_type = args.run_name # Set model type
unet_path = args.output_name+"_"+args.model_type+"_"+str(num_epochs)+"epochs"+"_"+str(len(film_dataset))+"images"+".pth" 

def reset_grad():
    optimizer.zero_grad()


if WANDB:
    wandb.init(
        project="tfg-segmentador_definitius",
        name=args.run_name + "_" + args.model_type + "_" + str(num_epochs) + "epochs", 
        config={
        "learning_rate": lr,
        "batch_size":BATCH_SIZE,
        "epochs": num_epochs,
        "architecture": args.model_type,
        "dataset": args.dataset,
        "dataset_size": len(film_dataset),
        }
    )

if NOTIFICATIONS:
    send_notification(f"Starting training of {args.model_type} 🚀")

for epoch in range(num_epochs):

    unet.train()
    epoch_loss = 0
    acc = 0
    length = 0

    for images, GT in train_loader:
        images, GT = images.to(device), GT.to(device)



        SR = unet(images)
        SR_probs = F.sigmoid(SR)
        SR_flat = SR_probs.view(SR_probs.size(0), -1)
        GT_flat = GT.view(GT.size(0), -1)

        loss = criterion(SR_flat, GT_flat)
        epoch_loss += loss.item()

        # Backpropagation
        reset_grad()
        loss.backward()
        optimizer.step()

        # Compute metrics
        acc += get_accuracy(SR, GT)
        length += images.size(0)

    acc= acc / length

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, \n'
          f'[Training] Acc: {acc:.4f}')

    gt_np = GT[0].cpu().detach().numpy().transpose(1, 2, 0)
    sr_np = SR_probs[0].cpu().detach().numpy().transpose(1, 2, 0)
    # Ensure images are 3-channel
    gt_np = np.repeat(gt_np, 3, axis=-1)
    sr_np = np.repeat(sr_np, 3, axis=-1)

    # Convert to grayscale
    gt_gray = np.mean(gt_np, axis=-1)
    sr_gray = np.mean(sr_np, axis=-1)

    # Define black pixel threshold
    black_thresh = 0.1

    # Create masks
    gt_black_mask = gt_gray < black_thresh
    sr_black_mask = sr_gray < black_thresh

    # Transparency level
    alpha = 0.5  

    # Create RGB masks with correct coloring
    gt_colored = np.zeros_like(gt_np)
    sr_colored = np.zeros_like(sr_np)

    # Blue for GT black pixels
    gt_colored[..., 2] = gt_black_mask * alpha  # Blue channel

    # Red for SR black pixels
    sr_colored[..., 0] = sr_black_mask * alpha  # Red channel

    # Blend with original images
    final_image = np.clip(gt_np + gt_colored + sr_colored, 0, 1)
    
    if WANDB:
        wandb.log({"epoch": epoch, "train_loss": epoch_loss, 
        "images": [
        wandb.Image(images[0].cpu().detach().numpy().transpose(1, 2, 0)),
        wandb.Image(final_image),
        wandb.Image(SR_probs[0].cpu().detach().numpy().transpose(1, 2, 0))
    ]})

    # Decay learning rate
    if (epoch + 1) > (num_epochs - num_epochs_decay):
        lr -= (lr / float(num_epochs_decay))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f'Decay learning rate to lr: {lr}.')

    # Validation
    unet.eval()

    length = 0
    acc = 0

    with torch.no_grad():
        for images, GT in valid_loader:
            images, GT = images.to(device), GT.to(device)
            SR = F.sigmoid(unet(images))
            val_loss = criterion(SR.view(SR.size(0), -1), GT.view(GT.size(0), -1))
            acc += get_accuracy(SR, GT)
            length += images.size(0)

    acc = acc / length
    gt_np = GT[0].cpu().detach().numpy().transpose(1, 2, 0)
    sr_np = SR[0].cpu().detach().numpy().transpose(1, 2, 0)
       # Ensure images are 3-channel
    gt_np = np.repeat(gt_np, 3, axis=-1)
    sr_np = np.repeat(sr_np, 3, axis=-1)

    # Convert to grayscale
    gt_gray = np.mean(gt_np, axis=-1)
    sr_gray = np.mean(sr_np, axis=-1)

    # Define black pixel threshold
    black_thresh = 0.1

    # Create masks
    gt_black_mask = gt_gray < black_thresh
    sr_black_mask = sr_gray < black_thresh

    # Transparency level
    alpha = 0.5  

    # Create RGB masks with correct coloring
    gt_colored = np.zeros_like(gt_np)
    sr_colored = np.zeros_like(sr_np)

    # Blue for GT black pixels
    gt_colored[..., 2] = gt_black_mask * alpha  # Blue channel

    # Red for SR black pixels
    sr_colored[..., 0] = sr_black_mask * alpha  # Red channel

    # Blend with original images
    final_image = np.clip(gt_np + gt_colored + sr_colored, 0, 1)
    
    print(f'[Validation] Acc: {acc:.4f}')
    if WANDB:
        wandb.log({ "val_acc": acc, "val_loss": val_loss, "epoch": epoch, "images_val": [
        wandb.Image(images[0].cpu().detach().numpy().transpose(1, 2, 0)),
        wandb.Image(final_image),
        wandb.Image(SR[0].cpu().detach().numpy().transpose(1, 2, 0))
        ]})



unet.to(device)
model_name = model_name.split("/")[-1].split(".")[0]

test_images = os.listdir(root_path + "/test")

dataset_test = TestDataset(root_path+"/test", limit=None, new_loader=new_loader)



for index, (image, img_path) in enumerate(dataset_test):
    #convert image tensor to numpy array
    image = image.permute(1, 2, 0).numpy()
    print(img_path)
    pred_mask = unet(torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device))
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
    break

model = unet.to("cpu")
torch.save(model.state_dict(), unet_path)

if NOTIFICATIONS:
    send_notification(f"Training finished of {args.model_type}  🎉")

if WANDB:
    wandb.finish()

import gc
gc.collect()
torch.cuda.empty_cache()
