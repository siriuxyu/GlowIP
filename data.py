import os
import numpy as np

data_path = "./data/test_data"

datasets = ["BraTS", "LDCT", "LIDC_320", "LIDC_512"]

def load_data(dataset):
    """
    Load data from the specified dataset directory.
    """
    file = os.path.join(data_path, dataset, ".npz")
    if os.path.exists(file):
        data = np.load(file)
        print(data)
        return data['arr_0']
    else:
        raise FileNotFoundError(f"Data file {file} not found.")

    
    
    