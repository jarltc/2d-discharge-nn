""" Python module for functions and stuff that have to do with autoencoder models. 

created by jarl on 31 Jul 2023 22:20
"""

import cv2
import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

nc_data = Path('/Users/jarl/2d-discharge-nn/data/interpolation_datasets/full_interpolation.nc')

def get_dataset_old(V, P, data_dir):
    """(old) function to load data from a pair of V, P

    Loads data from npz files.

    Args:
        V (int or float): voltage
        P (int or float): pressure

    Returns:
        np.array: Array with shape (5, 707, 200)
    """
    V = str(float(V))
    P = str(float(P))
    file = next(data_dir.glob(f'{V}_{P}.npz'))
    array = np.load(file)['arr_0']

    return array


def get_dataset(V, P, dataset:Path = nc_data):
    """Load the dataset for a pair of V, P from an nc file.

    Uses interpolation_datasets/full_interpolation.nc by default, and converts nans to zeros.

    Args:
        V (numeric): voltage
        P (numeric): pressure
        dataset (Path, optional): Path to .nc file to be used. 
            Defaults to Path('/Users/jarl/2d-discharge-nn/data/\ interpolation_datasets/full_interpolation.nc').

    Returns:
        np.ndarray: Dataset for the specified V, P pair, with shape (channels, height, width).
    """

    ds = xr.open_dataset(dataset)
    
    # each element in the array has shape (height, width), stacking gives (channels, height, width)
    data_array = np.stack([np.nan_to_num(ds[variable].sel(V=V, P=P).values) for variable in list(ds.keys())])

    return data_array


def crop(image:np.ndarray, corner:tuple =(0, 350), width:int =200, height:int =200):
    # image shape = 5, 707, 200
    """Crop images to desired size (width and height). 

    Args:
        image (np.ndarray): Image (channels, height, width) to be cropped.
        corner (Tuple[int, int]): Location of the lower-left corner for the cropping.
        width (Crop width): Width of the crop.
        height (Crop height): Height of the crop.

    Returns:
        np.ndarray: Cropped n-channel image with shape (n, height, width).
    """    
    startx, starty = corner

    endx = startx + width
    endy = starty + height

    return image[:, starty:endy, startx:endx]


def downscale(image_stack: np.ndarray, resolution: int) -> np.ndarray:
    """Downscale input images to lower resolution.

    Args:
        image_stack (np.ndarray): n-channel image (n, height, width) to downscale.
        resolution (int): Resolution of downscaled images.

    Returns:
        np.ndarray: Downscaled image with shape (channels, height, width).
    """
    
    data = np.stack([cv2.resize((np.moveaxis(image, 0, -1)), (resolution, resolution)) for image in image_stack])
    return np.moveaxis(data, -1, 1)  # revert moveaxis operation


def minmax_scale(image:np.ndarray, ds:xr.Dataset):
    """Perform minmax scaling on some input image with shape (channels, height, width)

    Values for the minmax scaling are obtained from the .nc file. 
    This also forces the minimum to be 0 instead of some crazy value that might mess with calculations.
    Args:
        image (np.ndarray): Image (channels, height, width) to be scaled.

    Returns:
        np.ndarray: Minmax-scaled array.
    """

    scaled_arrays = []

    for i, variable in enumerate(list(ds.keys())):

        var_data = np.nan_to_num(ds[variable].values)
        a = 0.0  # force 0 minimum
        b = var_data.max()

        data_array = image[i, :, :]

        # minmax scale
        scaled_arrays.append((data_array-a) / (b - a))

    scaled = np.stack(scaled_arrays)

    assert scaled.shape == image.shape

    return scaled


def get_data(test:tuple, validation:tuple = None, resolution=None, square=False):
    """Get train, test, and (optional) validation data from an .nc file.

    Assumes that test and validation sets are only single images. (This might change with a much larger dataset)
    Args:
        test (tuple): Pair of V, P for the test set.
        validation (tuple, optional): Pair of V, P for the validation set. Defaults to None.
        resolution (int, optional): Perform downscaling if specified. Defaults to None.
        square (bool, optional): Crops to a square if True. Defaults to False.

    Returns:
        [train, test, [validation]]: Minmax-scaled training and test images (np.ndarrays), and validation image if provided.
    """
    
    global nc_data
    ds = xr.open_dataset(nc_data)
    v_list = list(ds.V.values)
    p_list = list(ds.P.values)

    vps = [(v, p) for v in v_list for p in p_list]

    train_images = []
    for vp in vps:
        image = get_dataset(vp[0], vp[1])  # load data
        image = crop(image) if square else None
        image = downscale(image, resolution) if resolution is not None else None

        image = minmax_scale(image, ds)

        if vp == test:
            test_image = np.expand_dims(image, axis=0)
        elif vp == validation:
            val_image = np.expand_dims(image, axis=0)
        else:
            train_images.append(image)

        train_set = np.stack(train_images)

    if validation is not None:
        return [train_set, test_image, val_image]
    else:
        return [train_set, test_image]


class AugmentationDataset(Dataset):
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