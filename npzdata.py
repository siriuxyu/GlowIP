import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

path_remote = "./data/test_data"
path_local  = "/Users/siriux/Downloads/test_data"

data_path = path_remote if os.path.exists(path_remote) else path_local

datasets = ["BraTS", "LDCT", "LIDC_320", "LIDC_512"]


class NPZDataset(Dataset):
    def __init__(self, npz_file_path):
        data = np.load(npz_file_path)
        self.images = data['all_imgs']
        self.length = len(self.images)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image = self.images[idx]
        image = image.astype(np.float32) / 255.0
        return torch.from_numpy(image)

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
    data = load_data("LDCT")
    print(data['all_imgs'].shape)
    single_slice_0 = data['all_imgs'][1]
    single_slice_1 = data['all_imgs'][2]
    plt.imsave("test0.png", single_slice_0, cmap='gray')
    plt.imsave("test1.png", single_slice_1, cmap='gray')
    