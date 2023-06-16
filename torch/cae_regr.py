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

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = torchvision.transforms.functional.crop(
            decoded, 0, 0, 64, 64)
        return decoded


class MLP(nn.Module):
    """MLP to recreate encodings from a pair of V and P.
    """
    def __init__(self, input_size, output_size, dropout_prob) -> None:
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, output_size)  
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc4(x)
        x = F.relu(x)

        return x


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

if __name__ == '__main__':
    # set metal backend (apple socs)
    device = torch.device(
        'mps' if torch.backends.mps.is_available() else 'cpu')

    ae_dir = Path('/Users/jarl/2d-discharge-nn/created_models/autoencoder/32x32/A212/A212')
    mlp_dir = Path(input("Enter model directory (MLP): "))
    root = Path.cwd()
    is_square=True

    out_dir = mlp_dir.parents[0]
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    resolution = 32
    image_ds = ImageDataset(root/'data'/'interpolation_datasets', is_square)
    test_features, test_labels = image_ds.test
    test_res = resize(test_features, resolution)

    # scaler = MinMaxScaler()
    # scaled_labels = scaler.fit_transform(train_labels)
    # scaled_labels_test = scaler.transform(test_labels.reshape(1, -1))  # check if scaled first

    encodedx = 20
    encodedy = 5
    encodedz = 5
    encoded_size = encodedx*encodedy*encodedz

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
        fake_encoding = mlp(torch.tensor(test_labels, device=device, dtype=torch.float32))  # mps does not support float64
        decoded = model.decoder(fake_encoding)

    eval_time, scores = plot_comparison_ae(test_res, fake_encoding, model, out_dir=out_dir, is_square=is_square)
    r2 = ae_correlation(test_res, fake_encoding, out_dir)
