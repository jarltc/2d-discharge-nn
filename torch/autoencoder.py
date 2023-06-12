"""Autoencoder for predicting 2d plasma profiles

Jupyter eats up massive RAM so I'm making a script to do my tests
"""

import os
import time
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

from data_helpers import ImageDataset
from plot import plot_comparison_ae

# define model TODO: construct following input file/specification list

class SquareAE32(nn.Module):
    """Autoencoder using square images as inputs.
    
    Input sizes are (5, 32, 32) (no).
    """
    def __init__(self) -> None:
        super(SquareAE32, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(5, 10, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(10, 20, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(20, 20, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(20, 20, kernel_size=2, stride=2),
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


class Trial():  # TODO: add plotting
    def __init__(self, epochs, learning_rate, kernel1, kernel2) -> None:
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.model = Autoencoder()
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.epoch_loss = []

    def train(self):
        for epoch in self.epochs:
            for i, batch_data in enumerate(trainloader):
                inputs = batch_data[0]
                self.optimizer.zero_grad()

                # record loss
                running_loss = 0.0

                outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            self.epoch_loss.append(running_loss)


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


def plot_train_loss(losses, validation_losses=None):  # TODO: move to plot module

    losses = np.array(losses)
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.plot(losses, c='r', label='train')

    if validation_losses is not None:
        ax.plot(validation_losses, c='r', ls=':', label='validation')
        ax.legend()

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid()

    fig.savefig(out_dir/'train_loss.png')


def write_metadata(out_dir):  # TODO: move to data module
    # if is_square:
    #     in_size = (1, 5, 200, 200)
    # else:
    #     in_size = (1, 5, 707, 200)

    in_size = batch_data[0].size()  # testing

    # save model structure
    file = out_dir/'train_log.txt'
    with open(file, 'w') as f:
        f.write(f'Model {name}\n')
        print(summary(model, input_size=in_size), file=f)
        print("\n", file=f)
        print(model, file=f)
        f.write(f'\nEpochs: {epochs}\n')
        f.write(f'Learning rate: {learning_rate}\n')
        f.write(f'Resolution: {resolution}\n')
        f.write(f'Train time: {(train_end-train_start):.2f} s\n')
        # f.write(
        #     f'Average time per epoch: {np.array(epoch_times).mean():.2f} s\n')
        f.write(f'Evaluation time: {(eval_time):.2f} s\n')
        f.write(f'Scores (MSE): {scores}\n')
        f.write('\n***** end of file *****')


if __name__ == '__main__':
    # set metal backend (apple socs)
    device = torch.device(
        'mps' if torch.backends.mps.is_available() else 'cpu')

    name = input("Enter model name: ")
    root = Path.cwd()
    is_square=True

    image_ds = ImageDataset(root/'data'/'interpolation_datasets', is_square)
    train = image_ds.train[0]  # import only features (2d profiles)
    test = image_ds.test[0]  # import only features (2d profiles)

    # downscale train images
    resolution = 64
    train_res = resize(train, resolution)
    test_res = resize(test, resolution)

    # split validation set
    train_res, val = train_test_split(train_res, test_size=1, train_size=30)
    val = torch.tensor(val, device=device)

    out_dir = root/'created_models'/'autoencoder'/'64x64'/name
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    dataset = TensorDataset(torch.tensor(train_res, device=device))
    trainloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # hyperparameters (class property?)
    epochs = 200
    learning_rate = 1e-3

    if is_square:
        if resolution == 32:
            model = SquareAE32()
        elif resolution == 64:
            model = SquareAE64()
        # elif resolution == 200:
        #     model = SquareAE()
        else:
            raise ValueError(f"Resolution {resolution} is invalid!")
    else:
        model = Autoencoder()  # use full resolution model

    model.to(device)  # move model to gpu
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # include as class property?
    # weights_np = np.expand_dims(
    #     np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32), axis=(0, 2, 3))
    # weights = torch.tensor(weights_np, device=device)

    # convert to class method?
    epoch_loss = []
    epoch_validation = []
    epoch_times = []
    loop = tqdm(range(epochs))

    train_start = time.time()
    for epoch in loop:
        # loop = tqdm(trainloader)
        epoch_start = time.time()
        for i, batch_data in enumerate(trainloader):
            # get inputs
            inputs = batch_data[0]
            optimizer.zero_grad()

            # record loss
            running_loss = 0.0

            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_description(f"Epoch {epoch+1}/{epochs}")

        with torch.no_grad():
            val_loss = criterion(model(val), val).item()

        epoch_validation.append(val_loss)
        epoch_loss.append(running_loss)
        epoch_end = time.time()
        epoch_times.append(time.time()-epoch_start)

    train_end = time.time()

    prediction = model.encoder(test_res)

    torch.save(model.state_dict(), out_dir/f'{name}')
    eval_time, scores = plot_comparison_ae(test_res, prediction, out_dir=out_dir, is_square=is_square)
    plot_train_loss(epoch_loss, epoch_validation)
    write_metadata(out_dir)
