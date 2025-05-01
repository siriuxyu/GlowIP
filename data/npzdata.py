import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import cv2

path_remote = "./data/mri"
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

def load_data(dataset):
    """
    Load data from the specified dataset directory.
    """
    file = os.path.join(data_path, f'{dataset}.npz')
    if os.path.exists(file):
        data = np.load(file)
        return data
    else:
        raise FileNotFoundError(f"Data file {file} not found.")

    
if __name__ == "__main__":
    # data = load_data("LDCT")
    # print(data['all_imgs'].shape)
    # single_slice_0 = data['all_imgs'][1]
    # single_slice_1 = data['all_imgs'][2]
    # plt.imsave("test0.png", single_slice_0, cmap='gray')
    # plt.imsave("test1.png", single_slice_1, cmap='gray')
    
    dataset    = NPZDataset(path_local)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=100,
                                                drop_last=True, shuffle=True)
    for j, data in enumerate(dataloader):
            # loading batch
            x = data * 255.0
            print(x.shape)
