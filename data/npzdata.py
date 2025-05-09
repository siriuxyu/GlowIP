import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import argparse

path_remote = "./test_data/"
path_local  = "/Users/siriux/Downloads/mri_test_data/LDCT.npz"

data_path = path_remote if os.path.exists(path_remote) else path_local

datasets = ["BraTS", "LDCT", "LIDC_320", "LIDC_512"]


class NPZDataset(Dataset):
    def __init__(self, npz_file_path, size=64):
        data = np.load(npz_file_path)

        self.images = data['all_imgs']
        self.length = len(self.images)
        self.size = size
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = self.images[idx]
        if img.ndim == 2:
            # GrayScale: H x W → 1 x H x W
            img = img[np.newaxis, :, :]
        elif img.ndim == 3 and img.shape[0] not in (1, 3):
            # Color: C x H x W → 1 x C x H x W
            img = np.transpose(img, (2, 0, 1))
            
        img_resized = np.zeros((img.shape[0], self.size, self.size), dtype=np.float32)
        for c in range(img.shape[0]):
            img_resized[c] = cv2.resize(img[c], (self.size, self.size), interpolation=cv2.INTER_AREA)

        return torch.from_numpy(img_resized / 255.0).float()

    def sample_images(self, num_samples=1000):
        """
        Randomly sample a specified number of images from the dataset.
        """
        if num_samples <= 0:
            raise ValueError("num_samples must be a positive integer.")
        elif num_samples > self.length:
            num_samples = self.length
            return
        else:
            indices = np.random.choice(self.length, num_samples, replace=False)
            sampled_images = self.images[indices]
            self.images = sampled_images
            self.length = num_samples
            
def split_data(dataset, dir, train_ratio=0.8):
    """
    Split the dataset into training and testing sets.
    """
    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be between 0 and 1.")
    
    data = np.load(os.path.join(dir, f'{dataset}.npz'))
    images = data['all_imgs']
    
    length = len(images)
    indices = np.arange(length)
    np.random.shuffle(indices)
    split_index = int(length * train_ratio)
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]
    
    train_images = images[train_indices]
    test_images = images[test_indices]
    train_path = os.path.join(dir, f'{dataset}_train.npz')
    test_path = os.path.join(dir, f'{dataset}_test.npz')
    
    np.savez(train_path, all_imgs=train_images)
    np.savez(test_path, all_imgs=test_images)


    
if __name__ == "__main__":
    for dataset in datasets:
        path = os.path.join(data_path, f'{dataset}.npz')
        save_path = os.path.join(data_path, f'{dataset}/')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # copy from path to save_path
        os.system(f"cp {path} {save_path}")
        split_data(dataset, save_path)
        
        
