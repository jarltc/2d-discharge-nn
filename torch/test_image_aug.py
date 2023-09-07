""" Test loading of tensors without loading a whole pytorch tensor into memory """

import time
import xarray as xr
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import numpy as np
from image_data_helpers import crop
import matplotlib.pyplot as plt

ncfile = Path('/Users/jarl/2d-discharge-nn/data/interpolation_datasets/synthetic_averaged.nc')
ds = xr.open_dataset(ncfile, chunks={'images':62})

class CustomDataset(Dataset):
    def __init__(self, ncds, square=True):
        super().__init__()
        self.data = ncds
        self.is_square = square
    
    def __len__(self):
        return self.data.dims['image']
    
    def __getitem__(self, index):
        """ Convert batches of data from xarray to numpy arrays then pytorch tensors """
        # does the dataloader keep track of what data has already been used by tracking the indices?
        np_arrays = [self.data[variable].sel(image=index).values for variable in list(self.data.keys())]  # extract numpy array in each variable
        image_sample = np.stack(np_arrays)  # stack arrays into shape (channels, height, width)
        sample_tensor = torch.tensor(image_sample)  # convert to pytorch tensor
        if self.is_square:
            sample_tensor = crop(sample_tensor)  # crop to a square
        return sample_tensor


start = time.perf_counter_ns()
customdataset = CustomDataset(ds)
end = time.perf_counter_ns()
sample = customdataset[12]  # load a random image

print(f'data loaded in {(end-start)*1e-6} ms')

dataloader = DataLoader(customdataset, batch_size=32, shuffle=True)
data_iter = iter(dataloader)  # initialize an iterator object for the DataLoader
batch = next(data_iter)
print(batch.size())
 # sys.getsizeof() returns the size of an object in bytes