import os
import torch
import numpy as np
import csv

from torch.utils.data import Dataset
from torchvision import transforms

from skimage.io import imread
from skimage.transform import resize
from skimage.util import random_noise
from skimage.morphology import dilation, disk
from skimage.filters import gaussian
from skimage import img_as_ubyte
import random

class FilmDataset(Dataset):
    def __init__(self, root_path, height=1024, width=1024, limit=None, seed=42, export_csv=True, csv_path="image_mask_map.csv", new_method=False, sigma=1, white=0.0):
        self.root_path = root_path
        self.limit = limit
        self.height = height
        self.width = width
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.csv_path = csv_path
        self.sigma = sigma
        self.white = white

        # Load and sort images
        self.images = sorted([os.path.join(root_path, "train", i) for i in os.listdir(os.path.join(root_path, "train"))])[:self.limit]
        if self.limit is None:
            self.limit = len(self.images)
        self.masks = sorted([os.path.join(root_path, "train_mask", i) for i in os.listdir(os.path.join(root_path, "train_mask"))])[:self.limit]

        if self.limit is None:
            self.limit = len(self.images)

        # Ensure reproducibility
        random.seed(seed)
        shuffled_masks = self.masks.copy()
        random.shuffle(shuffled_masks)

        # Store image-mask pairing
        self.image_to_mask_map = {img: mask for img, mask in zip(self.images, shuffled_masks)}
        # Export to CSV if enabled
        if export_csv:
            self.export_mapping_to_csv()
        self.new_method = new_method

    def export_mapping_to_csv(self):
        """Exports image-mask mapping to a CSV file."""
        with open(self.csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Image", "Mask"])
            for img, mask in self.image_to_mask_map.items():
                writer.writerow([img, mask])

    def __getitem__(self, index):
        img_path = self.images[index]
        mask_path = self.image_to_mask_map[img_path]


        img = imread(img_path)  # Load RGB image
        mask = imread(mask_path)  # Load grayscale mask

        mask_height, mask_width = mask.shape
        img = img[:mask_height, :mask_width, :]

        mask_converted = mask.astype(np.float32) / 255.0
        if len(mask_converted.shape) == 2:  
            darkening_mask_colored = np.stack([mask_converted] * 3, axis=-1)
        else:
            darkening_mask_colored = mask_converted
        
        use_white = random.random() < self.white


        if not self.new_method:
            if use_white:
                img_dirty = img * darkening_mask_colored + 255 * (1 - darkening_mask_colored)
                img_dirty = img_dirty.astype(np.uint8)
            else:
                img_dirty = (img * darkening_mask_colored).astype(np.uint8)

            img_dirty_resized = resize(img_dirty, (self.height, self.width), mode='reflect', anti_aliasing=True)
            mask_resized = resize(mask, (self.height, self.width), mode='reflect', anti_aliasing=True)
            mask_resized = np.where(mask_resized > 0.9, 1, 0)

            
            img_tensor = torch.tensor(img_dirty_resized).permute(2, 0, 1).float()
            mask_tensor = torch.tensor(mask_resized).unsqueeze(0).float()
            return img_tensor, mask_tensor
        else:
            img_dirty = (img * darkening_mask_colored).astype(np.uint8)
            img_dirty_resized = resize(img_dirty, (self.height, self.width), mode='reflect', anti_aliasing=True)

            mask_resized = resize(mask, (self.height, self.width), mode='reflect', anti_aliasing=True)
            mask_resized_binary = np.where(mask_resized > 0.9, 1, 0)
            # mask_resized = np.where(mask_resized > 0.9, 1, 0)
            # Convert skimage mask to uint8 if needed
            mask_resized_uint8 = img_as_ubyte(mask_resized > 0)  # Ensure binary mask is uint8

            # Step 1: Dilate the mask to expand the dirt spots (10-pixel radius)
            # selem = disk(10)  # Structuring element of radius ~10
            # mask_dilated = dilation(mask_resized, selem).astype(np.uint8)  # Skimage dilation

            # Step 2: Apply Gaussian blur to the expanded region
            if img_dirty.ndim == 3:  # Check if it's a color image
                blurred_img = np.stack([gaussian(img_dirty_resized[..., i], sigma=self.sigma) for i in range(img_dirty_resized.shape[-1])], axis=-1)
            else:
                blurred_img = gaussian(img_dirty, sigma=5)  # Grayscale case

# Convert back to uint8
            blurred_img = img_as_ubyte(blurred_img)
            # Step 3: Superimpose the blurred dirt spots only where mask_resized was
            img_dirty_final = img_dirty_resized.copy()
            img_dirty_final[mask_resized_uint8 > 0] = blurred_img[mask_resized_uint8 > 0]
            
            img_tensor = torch.tensor(img_dirty_final).permute(2, 0, 1).float()
            mask_tensor = torch.tensor(mask_resized_binary).unsqueeze(0).float()
        
        return img_tensor, mask_tensor



    def __len__(self):
        return min(len(self.images), self.limit)


class FilmDatasetOther(Dataset):
    def __init__(self, root_path, height=1024, width=1024, limit=None, seed=42, export_csv=True, csv_path="image_mask_map.csv", new_method=False, sigma=1):
        self.root_path = root_path
        self.limit = limit
        self.height = height
        self.width = width
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.csv_path = csv_path
        self.sigma = sigma

        # Load and sort images
        self.images = sorted([os.path.join(root_path, "train", i) for i in os.listdir(os.path.join(root_path, "train"))])[:self.limit]
        if self.limit is None:
            self.limit = len(self.images)
        self.masks = sorted([os.path.join(root_path, "train_mask", i) for i in os.listdir(os.path.join(root_path, "train_mask"))])[:self.limit]

        if self.limit is None:
            self.limit = len(self.images)

        # Ensure reproducibility
        random.seed(seed)
        shuffled_masks = self.masks.copy()
        random.shuffle(shuffled_masks)

        # Store image-mask pairing
        self.image_to_mask_map = {img: mask for img, mask in zip(self.images, shuffled_masks)}
        # Export to CSV if enabled
        if export_csv:
            self.export_mapping_to_csv()
        self.new_method = new_method

    def export_mapping_to_csv(self):
        """Exports image-mask mapping to a CSV file."""
        with open(self.csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Image", "Mask"])
            for img, mask in self.image_to_mask_map.items():
                writer.writerow([img, mask])

    def __getitem__(self, index):
        img_path = self.images[index]
        mask_path = self.image_to_mask_map[img_path]


        img = imread(img_path)  # Load RGB image
        mask = imread(mask_path)  # Load grayscale mask

        mask_height, mask_width = mask.shape
        img = img[:mask_height, :mask_width, :]

        mask_converted = mask.astype(np.float32) / 255.0
        if len(mask_converted.shape) == 2:  
            darkening_mask_colored = np.stack([mask_converted] * 3, axis=-1)
        else:
            darkening_mask_colored = mask_converted
        if not self.new_method:
            img_dirty = (img * darkening_mask_colored).astype(np.uint8)

            img_dirty_resized = resize(img_dirty, (self.height, self.width), mode='reflect', anti_aliasing=True)
            mask_resized = resize(mask, (self.height, self.width), mode='reflect', anti_aliasing=True)
            mask_resized = np.where(mask_resized > 0.9, 1, 0)

            
            img_tensor = torch.tensor(img_dirty_resized).permute(2, 0, 1).float()
            mask_tensor = torch.tensor(mask_resized).unsqueeze(0).float()

            return img_tensor, mask_tensor
        else:
            img_dirty = (img * darkening_mask_colored).astype(np.uint8)
            img_dirty_resized = resize(img_dirty, (self.height, self.width), mode='reflect', anti_aliasing=True)

            mask_resized = resize(mask, (self.height, self.width), mode='reflect', anti_aliasing=True)
            mask_resized_binary = np.where(mask_resized > 0.9, 1, 0)
            # mask_resized = np.where(mask_resized > 0.9, 1, 0)
            # Convert skimage mask to uint8 if needed
            mask_resized_uint8 = img_as_ubyte(mask_resized > 0)  # Ensure binary mask is uint8

            # Step 1: Dilate the mask to expand the dirt spots (10-pixel radius)
            # selem = disk(10)  # Structuring element of radius ~10
            # mask_dilated = dilation(mask_resized, selem).astype(np.uint8)  # Skimage dilation

            # Step 2: Apply Gaussian blur to the expanded region
            if img_dirty.ndim == 3:  # Check if it's a color image
                blurred_img = np.stack([gaussian(img_dirty_resized[..., i], sigma=self.sigma) for i in range(img_dirty_resized.shape[-1])], axis=-1)
            else:
                blurred_img = gaussian(img_dirty, sigma=5)  # Grayscale case

# Convert back to uint8
            blurred_img = img_as_ubyte(blurred_img)
            # Step 3: Superimpose the blurred dirt spots only where mask_resized was
            img_dirty_final = img_dirty_resized.copy()
            img_dirty_final[mask_resized_uint8 > 0] = blurred_img[mask_resized_uint8 > 0]
            
            img_tensor = torch.tensor(img_dirty_final).permute(2, 0, 1).float()
            mask_tensor = torch.tensor(mask_resized_binary).unsqueeze(0).float()
        
        return img_tensor, mask_tensor



    def __len__(self):
        return min(len(self.images), self.limit)



class OldFilmDataset(Dataset):
    def __init__(self, root_path, limit=None):
        self.root_path = root_path
        self.limit = limit
        self.images = sorted([os.path.join(root_path, "train", i) for i in os.listdir(os.path.join(root_path, "train"))])[:self.limit]
        self.masks = sorted([os.path.join(root_path, "train_mask", i) for i in os.listdir(os.path.join(root_path, "train_mask"))])[:self.limit]

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        if self.limit is None:
            self.limit = len(self.images)

    def __getitem__(self, index):
        img = imread(self.images[index])  # Load RGB image
        mask = imread(self.masks[index], as_gray=True)  # Load grayscale mask

        img_resized = resize(img, (1024, 1024), mode='reflect', anti_aliasing=True)
        mask_resized = resize(mask, (1024, 1024), mode='reflect', anti_aliasing=True)
        mask_resized = np.where(mask_resized > 0.5, 1, 0)  # Binarize the mask

        img_tensor = torch.tensor(img_resized).permute(2, 0, 1).float()
        mask_tensor = torch.tensor(mask_resized).unsqueeze(0).float()

        return img_tensor, mask_tensor

    def __len__(self):
        return min(len(self.images), self.limit)

class TestDataset(Dataset):
    def __init__(self, root_path, limit=None, new_loader=False):
        self.root_path = root_path
        self.limit = limit
        self.images = sorted([os.path.join(root_path, "test_set", i) for i in os.listdir(os.path.join(root_path, "test_set"))])[:self.limit]
        if new_loader:
            self.images = sorted([os.path.join(root_path, "test_set_new", i) for i in os.listdir(os.path.join(root_path, "test_set_new"))])[:self.limit]

        # self.masks = sorted([os.path.join(root_path, "train_mask", i) for i in os.listdir(os.path.join(root_path, "train_mask"))])[:self.limit]

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        if self.limit is None:
            self.limit = len(self.images)

    def __getitem__(self, index):
        img = imread(self.images[index])  # Load RGB image

        img_resized = resize(img, (1024, 1024), mode='reflect', anti_aliasing=True)
        # mask_resized = resize(mask, (1024, 1024), mode='reflect', anti_aliasing=True)
        # mask_resized = np.where(mask_resized > 0.5, 1, 0)  # Binarize the mask

        img_tensor = torch.tensor(img_resized).permute(2, 0, 1).float()
        # mask_tensor = torch.tensor(mask_resized).unsqueeze(0).float()

        return img_tensor, self.images[index]

    def __len__(self):
        return min(len(self.images), self.limit)



class DenoisingDataset(Dataset):
    def __init__(self, root_path, height=1024, width=1024, limit=None, seed=42):
        self.root_path = root_path
        self.limit = limit
        self.height = height
        self.width = width
        self.transform = transforms.Compose([transforms.ToTensor()])

        # Load and sort images
        self.images = sorted([os.path.join(root_path, "train", i) for i in os.listdir(os.path.join(root_path, "train"))])[:self.limit]
        if self.limit is None:
            self.limit = len(self.images)


    def __getitem__(self, index):
        img_path = self.images[index]


        img = imread(img_path)  # Load RGB image
        img_resized = resize(img, (self.height, self.width), mode='reflect', anti_aliasing=True)
        # img_noisy_resized = random_noise(img_resized, mode='s&p', clip=True, amount=0.2)
        noise = np.random.normal(loc=0, scale=0.5, size=img_resized.shape[:2])  # Generate noise for one channel
        noise = np.expand_dims(noise, axis=-1)  # Add channel dimension
        noise = np.repeat(noise, img_resized.shape[-1], axis=-1)  # Repeat across all channels

        # Add noise and clip to valid range
        img_noisy_resized = np.clip(img_resized + noise, 0, 1)

        img_tensor = torch.tensor(img_resized).permute(2, 0, 1).float()
        noise_tensor = torch.tensor(img_noisy_resized).permute(2, 0, 1).float()

        return noise_tensor, img_tensor

    def __len__(self):
        return min(len(self.images), self.limit)


