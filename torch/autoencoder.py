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
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchinfo import summary

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from autoencoder_classes import A300
from data_helpers import ImageDataset, train2db
from plot import plot_comparison_ae, save_history_graph, ae_correlation


class Trial():  # TODO: add plotting
    def __init__(self, epochs, learning_rate, kernel1, kernel2) -> None:
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        # self.model = Autoencoder()
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

    name = input("Enter model name: ")
    root = Path.cwd()
    is_square=True

    # out_dir = root/'created_models'/'autoencoder'/'64x64'/name
    out_dir = root/'created_models'/'autoencoder'/'32x32'/name
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    image_ds = ImageDataset(root/'data'/'interpolation_datasets', is_square)
    train = image_ds.train[0]  # import only features (2d profiles)
    test = image_ds.test[0]  # import only features (2d profiles)

    # downscale train images
    resolution = 32
    train_res = resize(train, resolution)
    test_res = resize(test, resolution)

    train_res, scalers = normalize_train(train_res)
    test_res = normalize_test(test_res, scalers)

    with open(out_dir/'scalers.pkl', 'wb') as file:
        pickle.dump(scalers, file)
    file.close()

    # split validation set (takes one random set as validation)
    train_res, val = train_test_split(train_res, test_size=1, train_size=30)
    val = torch.tensor(val, device=device)

    dataset = TensorDataset(torch.tensor(train_res, device=device))
    trainloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # hyperparameters (class property?)
    epochs = 500
    learning_rate = 1e-3
    model = A300()
    model.to(device)  # move model to gpu
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # convert to class method?
    epoch_loss = []
    epoch_validation = []
    epoch_times = []
    loop = tqdm(range(epochs), desc='Training...', unit='epoch', colour='#7dc4e4')

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

        if (epoch+1) % epochs == 0:
            # save model every 10 epochs (so i dont lose all training progress in case i do something unwise)
            torch.save(model.state_dict(), out_dir/f'{name}')
            save_history_graph(epoch_loss, out_dir)

    train_end = time.time()

    with torch.no_grad():
        encoded = model.encoder(torch.tensor(test_res, device=device))
        decoded = model(torch.tensor(test_res, device=device)).cpu().numpy()

    torch.save(model.state_dict(), out_dir/f'{name}')
    train2db(out_dir, name, epochs, image_ds.v_excluded, image_ds.p_excluded, resolution, typ='autoencoder')
    eval_time, scores = plot_comparison_ae(test_res, encoded, model, out_dir=out_dir, is_square=is_square)
    r2 = ae_correlation(test_res, decoded, out_dir)
    plot_train_loss(epoch_loss, epoch_validation)
    write_metadata(out_dir)
