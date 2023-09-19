""" Test loading of tensors without loading a whole pytorch tensor into memory """

import time
import sys
from pathlib import Path
from tqdm import tqdm

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from autoencoder_classes import A64_6
from image_data_helpers import crop, downscale, get_data
from plot import plot_comparison_ae

ncfile = Path('/Users/jarl/2d-discharge-nn/data/interpolation_datasets/synthetic_averaged.nc')
ds = xr.open_dataset(ncfile, chunks={'images':62})
resolution = 64

class CustomDataset(Dataset):
    def __init__(self, directory, device, square=True, resolution=None):
        super().__init__()
        if resolution is not None:
            self.data = xr.open_dataset(directory/f'synthetic_averaged_s{resolution}.nc', chunks={'images': 62})
        else:
            self.data = xr.open_dataset(directory/'synthetic_averaged.nc', chunks={'images': 62})
        self.is_square = square
        self.resolution = resolution
        self.device = device
    
    def __len__(self):
        return self.data.dims['image']
    
    def __getitem__(self, index):
        """ Convert batches of data from xarray to numpy arrays then pytorch tensors """
        # does the dataloader keep track of what data has already been used by tracking the indices?
        np_arrays = [self.data[variable].sel(image=index).values for variable in list(self.data.keys())]  # extract numpy array in each variable
        image_sample = np.stack(np_arrays)  # stack arrays into shape (channels, height, width)
        tensor = torch.tensor(image_sample, device=self.device, dtype=torch.float32)  # convert to pytorch tensor
        return tensor
    
if __name__ == '__main__':
    ncfile = Path('/Users/jarl/2d-discharge-nn/data/interpolation_datasets/synthetic/synthetic_averaged.nc')
    ds = xr.open_dataset(ncfile, chunks={'images':62})
    data_dir = ncfile.parent
    resolution = 64

    device = torch.device('mps' if torch.backends.mps.is_available() 
                        else 'cpu')

    start = time.perf_counter_ns()
    customdataset = CustomDataset(data_dir, device, resolution=resolution)  # initialize dataset
    end = time.perf_counter_ns()
    sample = customdataset[12]  # load a random image

    print(f'data loaded in {(end-start)*1e-6} ms')

    dataloader = DataLoader(customdataset, batch_size=32, shuffle=True, num_workers=2)  # dataloader sends the data to the model
    # sys.getsizeof() returns the size of an object in bytes

    # hyperparameters
    hp_dict = {'epochs': 10, 
            'learning_rate': 1e-3, 
            'model': A64_6(), 
            'criterion': nn.MSELoss(),
            'images': len(customdataset)}

    epochs = hp_dict['epochs']
    learning_rate = hp_dict['learning_rate']
    model = hp_dict['model'].to(device)
    criterion = hp_dict['criterion']
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    hp_dict['optimizer'] = 'Adam'

    loop = tqdm(range(epochs), desc='Training...', unit='epoch', colour='#7dc4e4')

    train_start = time.perf_counter()
    for epoch in loop:
        for i, batch_data in enumerate(dataloader):
            # get inputs
            inputs = batch_data
            optimizer.zero_grad()

            # record loss
            running_loss = 0.0

            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_description(f'Epoch {epoch+1}/{epochs}')

    # training finishes
    torch.save(model.state_dict(), data_dir/'synthetic_test')
    train_end = time.perf_counter()
    write_metadata(data_dir)
    # load test set
    _, test_res = get_data((300, 60), 
                        resolution=customdataset.resolution, 
                        square=customdataset.is_square)
    model.eval()
    with torch.no_grad():
        encoded = model.encoder(torch.tensor(test_res, device=device, dtype=torch.float32))
        decoded = model(torch.tensor(test_res, device=device, dtype=torch.float32)).cpu().numpy()

    plot_comparison_ae(test_res, encoded, model, out_dir=data_dir, is_square=True, resolution=resolution)