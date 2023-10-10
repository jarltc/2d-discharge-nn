"""Autoencoder for predicting 2d plasma profiles

Jupyter eats up massive RAM so I'm making a script to do my tests
"""

import os
import time
import pickle
from pathlib import Path

import matplotlib.pyplot as plt

import cv2
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision.transforms.functional import crop
from torch.utils.data import TensorDataset, DataLoader
from torchinfo import summary

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from data_helpers import ImageDataset
from plot import plot_comparison_ae, save_history_graph, ae_correlation, image_slices
import autoencoder_classes
import mlp_classes
from image_data_helpers import get_data

# define model TODO: construct following input file/specification list

def resize(data: np.ndarray, scale=64) -> np.ndarray:
    """Resize square images to 64x64 resolution by downscaling.

    Args:
        data (np.ndarray): Input data.

    Returns:
        np.ndarray: Downscaled input data.
    """

    data = np.stack([cv2.resize((np.moveaxis(image, 0, -1)), (scale, scale)) for image in data])
    data = np.moveaxis(data, -1, 1)
    return data


def normalize_test(dataset:np.ndarray, scalers:dict()):
    normalized_variables = []

    for i, var in enumerate(['pot', 'ne', 'ni', 'nm', 'te']):
        x = dataset[:, i, :, :]
        xMin, xMax = scalers[var]
        scaledx = (x-xMin) / (xMax-xMin)  # shape: (31, x, x)
        normalized_variables.append(scaledx)
    
    # shape: (5, 31, x, x)
    normalized_dataset = np.moveaxis(np.stack(normalized_variables), 0, 1)  # shape: (31, 5, x, x)
    return normalized_dataset

if __name__ == '__main__':
    # set metal backend (apple socs)
    device = torch.device(
        'mps' if torch.backends.mps.is_available() else 'cpu')

    root = Path.cwd()
    is_square=True

    resolution = 64
    if resolution == 32:
        model = autoencoder_classes.A300()
        encodedx = 20
        encodedy = encodedz = 4
    elif resolution == 64:
        model = autoencoder_classes.A64_6()
        encodedx = 40
        encodedy = encodedz = 8
    encoded_size = encodedx * encodedy * encodedz
    # model = autoencoder_classes.A300()
    mlp = mlp_classes.MLP(2, encoded_size, dropout_prob=0.5)
    
    # ae_dir = Path(input('AE dir: '))
    ae_dir = Path('/Users/jarl/2d-discharge-nn/created_models/autoencoder/64x64/A64_6new/A64_6new')
    mlp_dir = Path('/Users/jarl/2d-discharge-nn/created_models/conditional_autoencoder/A64_new/A64_new')

    out_dir = mlp_dir.parents[0]
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    image_ds = ImageDataset(root/'data'/'interpolation_datasets', is_square)
    _, test = get_data((300, 60), resolution=resolution, square=is_square, labeled=True)
    test_image, test_label = test

    model.load_state_dict(torch.load(ae_dir))  # use path directly to model
    mlp.load_state_dict(torch.load(mlp_dir))  # use path directly to model
    model.to(device)  # move model to gpu
    mlp.to(device)
    model.eval()
    mlp.eval()

    
    with torch.no_grad():
        start = time.perf_counter_ns()
        fake_encoding = mlp(torch.tensor(test_label, device=device, dtype=torch.float32))  # mps does not support float64
        fake_encoding = fake_encoding.reshape(1, encodedx, encodedy, encodedz)
        decoded = model.decoder(fake_encoding).cpu().numpy()[:, :, :64, :64]
        end = time.perf_counter_ns()
    
    r2 = ae_correlation(test_image, decoded, out_dir)
    image_slices(test_image, decoded, out_dir=out_dir, cmap='viridis')
    delta(test_image, decoded, out_dir=out_dir, is_square=True)
    eval_time = (end-start) # ms
    write_metadata(out_dir)
