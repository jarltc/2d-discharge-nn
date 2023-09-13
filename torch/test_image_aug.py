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

from autoencoder_classes import A64_7
from image_data_helpers import crop, downscale

ncfile = Path('/Users/jarl/2d-discharge-nn/data/interpolation_datasets/synthetic_averaged.nc')
ds = xr.open_dataset(ncfile, chunks={'images':62})

class CustomDataset(Dataset):
    def __init__(self, ncds, device, square=True):
        super().__init__()
        self.data = ncds
        self.is_square = square
        self.device = device
    
    def __len__(self):
        return self.data.dims['image']
    
    def __getitem__(self, index):
        """ Convert batches of data from xarray to numpy arrays then pytorch tensors """
        # does the dataloader keep track of what data has already been used by tracking the indices?
        np_arrays = [self.data[variable].sel(image=index).values for variable in list(self.data.keys())]  # extract numpy array in each variable
        image_sample = np.stack(np_arrays)  # stack arrays into shape (channels, height, width)
        if self.is_square:
            tensor = crop(image_sample)  # crop to a square
            tensor = downscale(tensor, resolution=64)
        sample_tensor = torch.tensor(tensor, device=self.device, dtype=torch.float32)  # convert to pytorch tensor
        return sample_tensor

device = torch.device('mps' if torch.backends.mps.is_available() 
                      else 'cpu')

start = time.perf_counter_ns()
customdataset = CustomDataset(ds, device)
end = time.perf_counter_ns()
sample = customdataset[12]  # load a random image

print(f'data loaded in {(end-start)*1e-6} ms')

dataloader = DataLoader(customdataset, batch_size=32, shuffle=True)
# sys.getsizeof() returns the size of an object in bytes

# hyperparameters
epochs = 10
learning_rate = 1e-3
model = A64_7().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loop = tqdm(range(epochs), desc='Training...', unit='epoch', colour='#7dc4e4')

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