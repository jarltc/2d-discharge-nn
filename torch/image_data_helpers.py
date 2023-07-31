""" Python module for functions and stuff that have to do with autoencoder models. 

created by jarl on 31 Jul 22:20
"""

import cv2
import xarray as xr
import numpy as np
import torch
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

    Uses interpolation_datasets/full_interpolation.nc by default.

    Args:
        V (numeric): voltage
        P (numeric): pressure
        dataset (Path, optional): Path to .nc file to be used. 
            Defaults to Path('/Users/jarl/2d-discharge-nn/data/\ interpolation_datasets/full_interpolation.nc').

    Returns:
        _type_: _description_
    """

    ds = xr.open_dataset(dataset)
    
    # each element in the array has shape (height, width), stacking gives (channels, height, width)
    data_array = np.stack([np.nan_to_num(ds[variable].sel(V=V, P=P).values) for variable in list(ds.keys())])

    return data_array


def crop(image:np.ndarray, corner:tuple =(0, 350), width:int =200, height:int =200):
    # image shape = 5, 707, 200
    """Crop images to desired size (width and height). 

    Assumes input images with a shape of (channels, height, width).

    Args:
        image (np.ndarray): Image to be cropped.
        corner (Tuple[int, int]): Location of the lower-left corner for the cropping.
        width (Crop width): Width of the crop.
        height (Crop height): Height of the crop.

    Returns:
        np.ndarray: Cropped n-channel image.
    """    
    startx, starty = corner

    endx = startx + width
    endy = starty + height

    return image[:, starty:endy, startx:endx]


def downscale(image_stack: np.ndarray, resolution: int) -> np.ndarray:
    """Downscale input images to lower resolution.

    Args:
        image_stack (np.ndarray): n-channel image to downscale
        resolution (int): Resolution of downscaled images.

    Returns:
        np.ndarray: Downscaled image.
    """
    
    data = np.stack([cv2.resize((np.moveaxis(image, 0, -1)), (resolution, resolution)) for image in image_stack])
    return np.moveaxis(data, -1, 1)  # revert moveaxis operation


def minmax_scale(image:np.ndarray):
    """Perform minmax scaling on some input image with shape (channels, height, width)

    Args:
        image (np.ndarray): Image to be scaled, with shape (channels, height, width)

    Returns:
        np.ndarray: Minmax-scaled array.
    """

    global nc_data
    scaled_arrays = []

    for i, variable in enumerate(list(nc_data.keys())):

        var_data = np.nan_to_num(nc_data[variable].values)
        a = 0.0  # force 0 minimum
        b = var_data.max()

        data_array = image[i, :, :]

        # minmax scale
        scaled_arrays.append((data_array-a) / (b - a))

    scaled = np.stack(scaled_arrays)

    assert scaled.shape == image.shape

    return scaled

