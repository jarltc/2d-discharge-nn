""" Python module for functions and stuff that have to do with autoencoder models. 

created by jarl on 31 Jul 2023 22:20
"""

import cv2
import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import xbatcher

root = Path.cwd()
nc_data = root/'data'/'interpolation_datasets'/'full_interpolation.nc'
nc_data1 = root.parent/'data'/'interpolation_datasets'/'full_interpolation.nc'

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


def minmax_scale(image:np.ndarray, ds:xr.Dataset, max_value='true'):
    """Perform minmax scaling on some input image with shape (channels, height, width)

    Values for the minmax scaling are obtained from the .nc file. 
    This also forces the minimum to be 0 instead of some crazy value that might mess with calculations.
    This scheme scales an array to lie between 0 and 1 (the maximum value for the specific variable across the entire dataset).
    Args:
        image (np.ndarray): Image (channels, height, width) to be scaled.
        ds (xr.Dataset): Dataset to reference for the maximum values of each parameter.

    Returns:
        np.ndarray: Minmax-scaled array.
    """

    scaled_arrays = []

    if max_value not in ['true', '99', '999']:
        raise ValueError("Invalid max value. Supported schemes are 'true', '99' (99th percentile), and '999' (99.9th percentile).")

    def get_max(variable, minmax_scheme=max_value):
            # TODO: keep a dictionary of this data
            var_data = np.nan_to_num(ds[variable].values)
            if minmax_scheme == 'true':
                return var_data.max()  # use minmax values
            elif minmax_scheme == '999':
                return np.quantile(var_data, 0.999)
            elif minmax_scheme == '99':
                return np.quantile(var_data, 0.99)
            
    b = np.array([get_max(var) for var in list(ds.keys())]).reshape(5, 1, 1)  # reshape to allow broadcasting
    a = np.zeros((5, 1, 1))  # may change for whatever reason

    scaled = (image - a) / (b - a)  # minmax: x' = (x - min) / (max - min)

    return scaled


def minmax_label(V, P):
    """Get minmax-scaled V and P based on the available simulation dataset.

    Args:
        V (float): Voltage to be scaled.
        P (float): Pressure to be scaled.

    Returns:
        list of float: Scaled voltage and pressure.
    """
    global nc_data, ncdata1
    if nc_data.exists():
        ds = xr.open_dataset(nc_data)
    else:
        ds = xr.open_dataset(nc_data1)

    vs = ds.V.values
    ps = ds.P.values

    scaled_v = (vs.max() - V)/vs.max()
    scaled_p = (ps.max() - P)/ps.max()
    
    return [scaled_v, scaled_p]


def get_data(test:tuple, validation:tuple = None, resolution=None, square=True, labeled=False,
             minmax_scheme='true'):
    """Get train, test, and (optional) validation data from an .nc file.

    Assumes that test and validation sets are only single images. If labeled=True, return (minmax-scaled)
    Args:
        test (tuple): Pair of V, P for the test set.
        validation (tuple, optional): Pair of V, P for the validation set. Defaults to None.
        resolution (int, optional): Perform downscaling if specified. Defaults to None.
        square (bool, optional): Crops to a square if True. Defaults to False.
        labeled (bool, optional): Returns labeled data alongside the other requested data.
        minmax_scheme (str, optional): Select a minmax scheme. Supported schemes are true minmax ('true') or 99.9th percentile ('999') 
            Defaults to True.

    Returns:
        [train, test, *validation]: Minmax-scaled training and test images (np.ndarrays), and validation image if provided.
        [(train_images, train_labels), (test_image, test_label), *(val_set, val_label)] if labeled=True.
    """
    resolutions = [32, 64, 200, None]
    if resolution not in resolutions:
        raise ValueError(f'Invalid resolution: {resolution}. Expected one of : {resolutions}')

    global nc_data
    ds = xr.open_dataset(nc_data)
    v_list = list(ds.V.values)
    p_list = list(ds.P.values)

    vps = [(v, p) for v in v_list for p in p_list]

    train_images = []
    train_labels_list = []
    for vp in vps:
        image = get_dataset(vp[0], vp[1])  # load data
        cropped = crop(image) if square else image  # crop if square
        scaled = downscale(cropped, resolution) if resolution in [64, 32] else cropped  # downscale if 64, 32

        mmax = minmax_scale(scaled, ds, minmax_scheme)
        label = np.array(minmax_label(vp[0], vp[1]))  # shape = [2]

        if vp == test:
            test_image = np.expand_dims(mmax, axis=0)  # get the image and add an extra axis to match shape into (1, channels, width, height)
            test_label = np.expand_dims(label, axis=0)  # same thing here
        elif vp == validation:
            val_image = np.expand_dims(mmax, axis=0)
            val_label = np.expand_dims(label, axis=0)
        else:
            train_images.append(mmax)
            train_labels_list.append(label)
        
    train_set = np.stack(train_images) 
    train_labels = np.stack(train_labels_list)  # shape = (n_train_samples, 2)
    
    if not labeled:
        if validation is not None:
            print(f'loaded sim data with shapes {[data.shape for data in [train_set, test_image, val_image]]}')
            return [train_set, test_image, val_image]
        else:
            print(f'loaded sim data with shapes {[data.shape for data in [train_set, test_image]]}')
            return [train_set, test_image]
    else:
        if validation is not None:
            return [(train_set, train_labels), (test_image, test_label), (val_image, val_label)]
        else:
            return [(train_set, train_labels), (test_image, test_label)]


class AugmentationDataset(Dataset):
    def __init__(self, ncfile, device, dtype=torch.float32, is_square=False):
        super().__init__()
        self.data = xr.open_dataset(ncfile, chunks='auto')

        # self.resolution = resolution  # unused
        self.device = device
        self.dtype = dtype
        self.square = is_square
    
    def __len__(self):
        return self.data.dims['image']
    
    def _crop(np_array):
        return np_array.copy()[:, :, :, :200]
    
    def __getitem__(self, index):
        """ Convert batches of data from xarray to numpy arrays then pytorch tensors """
        # does the dataloader keep track of what data has already been used by tracking the indices?
        np_arrays = [self.data[variable].sel(image=index).values for variable in list(self.data.keys())]  # extract numpy array in each variable
        image_sample = self._crop(np.stack(np_arrays)) if self.square else np.stack(np_arrays) # stack arrays into shape (channels, height, width)
        tensor = torch.tensor(image_sample, device=self.device, dtype=self.dtype)  # convert to pytorch tensor
        self.data.close()
        return tensor
    

def check_empty(image_set:np.ndarray, eps=0.001):
    """ Check if a set of predictions contains an empty image.

    Args:
        image_set (np.ndarray): Image set to be checked.
        eps (float, optional): Threshold value to consider 
            if the image is empty or not. Defaults to 0.001.

    Returns:
        bool: True when the set contains an empty image. False otherwise.
    """
    means = np.array([image_set[:, i].mean() for i in range(5)])  # get the mean value for each channel
    return np.any(means < eps)  # check if any of these are less than eps
    