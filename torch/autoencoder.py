"""Autoencoder for predicting 2d plasma profiles

Jupyter eats up massive RAM so I'm making a script to do my tests
"""

import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as pat

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

# from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from plot import draw_apparatus

# define model TODO: construct following input file/specification list

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(5, 10, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(20, 40, kernel_size=5, stride=2, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(40, 20, kernel_size=3, stride=2),
            nn.ConvTranspose2d(20, 10, kernel_size=3, stride=2),
            nn.ConvTranspose2d(10, 5, kernel_size=3, stride=2)
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


def nc_to_train(ds):
    variables = list(ds.data_vars)
    v_excluded = 300  # TODO: move somewhere else
    p_excluded = 60

    data_list = []
    feature_list = []
    for v in ds.V.values:
        for p in ds.P.values:
            # extract the values from the dataset for all 5 variables
            vp_data = np.nan_to_num(np.stack(
                [scale_np(ds[var].sel(V=v, P=p).values, var, scaler_dict) for var in variables]))
            feature = np.array([v, p])
            if (v == v_excluded) & (p == p_excluded):
                # this is a hole in the data set that contains only nans
                pass
            else:
                data_list.append(vp_data)
                feature_list.append(feature)

    labels = np.float32(np.stack(data_list))
    features = np.float32(np.stack(data_list))  # not yet used
    # samples, channels, height, width
    assert labels.shape == (31, 5, 707, 200)

    torch.save(labels, train_data.parent/'train_set.pt')

    return train


def nc_to_test(ds):
    variables = list(ds.data_vars)

    for v in ds.V.values:
        for p in ds.P.values:
            # extract the values from the dataset for all 5 variables
            vp_data = np.nan_to_num(np.stack(
                [scale_np(ds[var].sel(V=v, P=p).values, var, scaler_dict) for var in variables]))
            features = np.array([v, p])  # not yet used

    # consider saving as .pt file after conversion
    labels = np.expand_dims(np.float32(vp_data), axis=0)
    assert labels.shape == (1, 5, 707, 200)  # samples, channels, height, width

    torch.save(labels, train_data.parent/'test_set.pt')

    return labels


def scale_np(array, var, scaler_dict):
    max = np.nanmax(array)
    min = np.nanmin(array)

    scaler_dict[var] = (min, max)

    return (array - min) / (max - min)


def plot_comparison_ae(reference: np.ndarray, name=None, out_dir=None):  # TODO: move to plot module
    """Create plot comparing the reference data with its autoencoder reconstruction.

    Args:
        reference (np.ndarray): Reference dataset.
        name (str, optional): Model name. Defaults to None.
        out_dir (Path, optional): Output directory. Defaults to None.

    Returns:
        _type_: _description_
    """
    fig = plt.figure(figsize=(12, 7), dpi=200)
    subfigs = fig.subfigures(nrows=2)

    axs1 = subfigs[0].subplots(nrows=1, ncols=5)
    axs2 = subfigs[1].subplots(nrows=1, ncols=5)

    subfigs[1].suptitle('Reconstruction')
    subfigs[0].suptitle('Original')

    cbar_ranges = [(reference[0, i, :, :].min(),
                    reference[0, i, :, :].max()) for i in range(5)]

    start = time.time()

    with torch.no_grad():
        encoded = model.encoder(torch.tensor(reference, device=device))
        reconstruction = model.decoder(encoded).cpu().numpy()

    end = time.time()

    for i in range(5):
        org = axs1[i].imshow(reference[0, i, :, :], origin='lower', extent=[
                             0, 20, 0, 70.7], cmap='Greys_r')
        draw_apparatus(axs1[i])
        plt.colorbar(org)
        rec = axs2[i].imshow(reconstruction[0, i, :, :], origin='lower', extent=[0, 20, 0, 70.7],
                             vmin=cbar_ranges[i][0], vmax=cbar_ranges[i][1], cmap='Greys_r')
        draw_apparatus(axs2[i])
        plt.colorbar(rec)

    if out_dir is not None:
        fig.savefig(out_dir/f'test_comparison.png')

    return end-start


def plot_train_loss(losses):  # TODO: move to plot module
    losses = np.array(losses)
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.plot(losses, c='r')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid()

    fig.savefig(out_dir/'train_loss.png')


def write_metadata(out_dir):  # TODO: move to data module
    # save model structure
    file = out_dir/'train_log.txt'
    with open(file, 'w') as f:
        f.write(f'Model {name}\n')
        print(summary(model, input_size=(1, 5, 707, 200)), file=f)
        f.write(f'\nEpochs: {epochs}\n')
        f.write(f'Learning rate: {learning_rate}\n')
        f.write(f'Execution time: {(train_end-train_start):.2f} s\n')
        f.write(
            f'Average time per epoch: {np.array(epoch_times).mean():.2f} s\n')
        f.write(f'Evaluation time: {(eval_time):.2f} s\n')
        f.write('\n***** end of file *****')


def load_image_data(root: Path, square=False):  # TODO: move to data module, convert to ImageDataset class
    """Load image data for image-based networks.

    Args:
        root (Path): Path to root 2d-discharge-nn folder.
        square (bool, optional): Crop the images to a square.
    """
    
    scaler_dict = {}  # TODO: use as class attribute

    train_data = root/'data'/'interpolation_datasets'/'train_set.pt'
    test_data = root/'data'/'interpolation_datasets'/'test_set.pt'

    if not train_data.exists():  # consider using a try-except
        train_data = root/'data'/'interpolation_datasets'/'rec-interpolation2.nc'
        train_ds = xr.open_dataset(train_data)
        train = nc_to_train(train_ds)
    else:
        train = torch.load(train_data)

    if not test_data.exists():
        test_data = root/'data'/'interpolation_datasets'/'test_set.nc'
        test_ds = xr.open_dataset(test_data)
        test = nc_to_test(test_ds)
    else:
        test = torch.load(test_data)

    if square:
        test = crop(test, 350, 0, 200, 200)  # (array, top, left, height, width)
        train = crop(test, 350, 0, 200, 200)  # (array, top, left, height, width)

    return train, test


if __name__ == '__main__':
    # set metal backend (apple socs)
    device = torch.device(
        'mps' if torch.backends.mps.is_available() else 'cpu')

    name = input("Enter model name: ")
    root = Path.cwd()

    train, test = load_image_data(root)

    out_dir = root/'created_models'/'autoencoder'/name
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    dataset = TensorDataset(torch.tensor(train, device=device))
    trainloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # hyperparameters (class property?)
    epochs = 200
    learning_rate = 1e-3

    model = Autoencoder()
    model.to(device)  # move model to gpu
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # include as class property?
    weights_np = np.expand_dims(
        np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32), axis=(0, 2, 3))
    weights = torch.tensor(weights_np, device=device)

    # convert to class method?
    epoch_loss = []
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
            loss = criterion(outputs * weights, inputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_description(f"Epoch {epoch+1}/{epochs}")

        epoch_loss.append(running_loss)
        epoch_end = time.time()
        epoch_times.append(time.time()-epoch_start)

    train_end = time.time()

    eval_time = plot_comparison_ae(test, out_dir=out_dir)
    plot_train_loss(epoch_loss)
    write_metadata(out_dir)
    torch.save(model.state_dict(), out_dir/f'{name}')
