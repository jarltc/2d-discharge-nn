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
from plot import plot_comparison_ae, ae_correlation, image_slices
from autoencoder_classes import A64_6
from image_data_helpers import get_data

# define model TODO: construct following input file/specification list

class SquareAE32(nn.Module):
    """Autoencoder using square images as inputs.
    
    Input sizes are (5, 32, 32) (no).
    """
    def __init__(self) -> None:
        super(SquareAE32, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(5, 10, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(20, 20, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(20, 20, kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(),

            nn.ConvTranspose2d(20, 10, kernel_size=2, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(10, 5, kernel_size=2, stride=2),
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        # decoded = torchvision.transforms.functional.crop(
        #     decoded, 0, 0, 64, 64)
        return decoded


class SquareAE64(nn.Module):
    """Autoencoder using square images as inputs.
    
    Input sizes are (5, 64, 64).
    """
    def __init__(self) -> None:
        super(SquareAE64, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(5, 10, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),

            nn.Conv2d(10, 20, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),

            nn.Conv2d(20, 40, kernel_size=1, stride=2, padding=0),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(40, 40, kernel_size=3, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(40, 20, kernel_size=5, stride=2),
            nn.ReLU(),

            nn.Conv2d(20, 20, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(20, 10, kernel_size=5, stride=2),
            nn.ReLU(),

            nn.Conv2d(10, 5, kernel_size=1, stride=1),
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = torchvision.transforms.functional.crop(
            decoded, 0, 0, 64, 64)
        return decoded


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(5, 10, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(20, 40, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(40, 20, kernel_size=3, stride=2),
            nn.ConvTranspose2d(20, 10, kernel_size=3, stride=2),
            nn.ConvTranspose2d(10, 5, kernel_size=5, stride=2)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = torchvision.transforms.functional.crop(
            decoded, 0, 0, 707, 200)
        return decoded


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


def write_metadata(out_dir):  # TODO: move to data module
    # if is_square:
    #     in_size = (1, 5, 200, 200)
    # else:
    #     in_size = (1, 5, 707, 200)

    # in_size = batch_data[0].size()  # testing
    in_size = (1, 5, 32, 32)  # change

    # save model structure
    file = out_dir/'train_log2.txt'
    with open(file, 'w') as f:
        # f.write(f'Model {name}\n')
        print(summary(model, input_size=in_size), file=f)
        print("\n", file=f)
        print(model, file=f)
        f.write(f'Resolution: {resolution}\n')
        # f.write(f'Train time: {(train_end-train_start):.2f} s\n')
        # f.write(
        #     f'Average time per epoch: {np.array(epoch_times).mean():.2f} s\n')
        f.write(f'Evaluation time: {(eval_time):.2f} s\n')
        f.write(f'Scores (MSE): {scores}\n')
        f.write(f'Scores (r2): {r2}\n')
        f.write('\n***** end of file *****')


def normalize_train(dataset:np.ndarray):
    normalized_variables = []
    scalers = {}

    for i, var in enumerate(['pot', 'ne', 'ni', 'nm', 'te']):
        x = dataset[:, i, :, :]
        xMax = np.max(x)
        xMin = np.min(x)
        scalers[var] = (xMin, xMax)
        scaledx = (x-xMin) / (xMax-xMin)  # shape: (31, x, x)
        normalized_variables.append(scaledx)
    # shape: (5, 31, x, x)
    normalized_dataset = np.moveaxis(np.stack(normalized_variables), 0, 1)  # shape: (31, 5, x, x)
    return normalized_dataset, scalers


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

    model_folder = Path(input("Enter model directory: "))
    model_dir = model_folder/model_folder.stem
    root = Path.cwd()
    is_square=True

    out_dir = model_folder
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    # downscale train images
    resolution = 64
    _, test_res = get_data((300, 60), resolution=resolution, square=True)
    
    model = A64_6()
    model.load_state_dict(torch.load(model_dir))  # use path directly to model
    model.to(device)  # move model to gpu
    model.eval()

    with torch.no_grad():
        encoded = model.encoder(torch.tensor(test_res, device=device, dtype=torch.float32))
        decoded = model(torch.tensor(test_res, device=device, dtype=torch.float32))

    eval_time, scores = plot_comparison_ae(test_res, encoded, model, 
                                           out_dir=out_dir, is_square=is_square, 
                                           cbar='viridis')
    r2 = ae_correlation(test_res, decoded, out_dir.parents[0])
    hslice, vslice, refplot = image_slices(test_res, decoded.cpu().numpy()[:,:,:resolution,:resolution], out_dir)
    print(f'plots [comparison, correlation, hslice, vslice, refplot] have been saved in {out_dir}')
