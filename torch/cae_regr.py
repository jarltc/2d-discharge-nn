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
from plot import plot_comparison_ae, save_history_graph, ae_correlation
import autoencoder_classes
import mlp_classes

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

    ae_dir = Path('/Users/jarl/2d-discharge-nn/created_models/autoencoder/32x32/A212/A212')
    mlp_dir = Path(input("Enter model directory (MLP): "))
    root = Path.cwd()
    is_square=True

    resolution = 64
    encodedx = 40
    encodedy = 8
    encodedz = 8
    encoded_size = encodedx*encodedy*encodedz
    model = autoencoder_classes.A64_6()
    # model = autoencoder_classes.A300()
    mlp = mlp_classes.MLP(2, encoded_size, dropout_prob=0.5)
    
    ae_dir = model.path
    mlp_dir = mlp.path64

    out_dir = mlp_dir.parents[0]
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    resolution = 32
    image_ds = ImageDataset(root/'data'/'interpolation_datasets', is_square)
    train_labels = image_ds.train[1]
    test_features, test_labels = image_ds.test
    test_res = resize(test_features, resolution)

    image_scalefile = ae_dir.parents[0]/'scalers.pkl'
    if image_scalefile.exists():
        with open(image_scalefile, 'rb') as f:
            imageScaler = pickle.load(f)
        test_res = normalize_test(test_res, imageScaler)

    scaler = MinMaxScaler()
    scaled_labels = scaler.fit_transform(train_labels)
    scaled_labels_test = scaler.transform(test_labels.reshape(1, -1))  # check if scaled first

    # TODO THE CODE DOESN'T WORK RIGHT BECAUSE I WAS SAVING THE WRONG MODEL THE WHOLE TIME (!!)
    model = SquareAE32()
    mlp = MLP(2, encoded_size, dropout_prob=0.5)
    mlp.load_state_dict(torch.load(mlp_dir))
    model.load_state_dict(torch.load(ae_dir))  # use path directly to model
    mlp.load_state_dict(torch.load(mlp_dir))  # use path directly to model
    model.to(device)  # move model to gpu
    mlp.to(device)
    model.eval()
    mlp.eval()

    with torch.no_grad():
        fake_encoding = mlp(torch.tensor(scaled_labels_test, device=device, dtype=torch.float32))  # mps does not support float64
        fake_encoding = fake_encoding.reshape(1, encodedx, encodedy, encodedz)
        decoded = model.decoder(fake_encoding)

    eval_time, scores = plot_comparison_ae(test_res, fake_encoding, model, out_dir=out_dir, is_square=is_square)
    r2 = ae_correlation(test_res, fake_encoding, out_dir)
