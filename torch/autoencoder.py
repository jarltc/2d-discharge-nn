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

from autoencoder_classes import A300, A64_7, A64_6s, A300s
from data_helpers import ImageDataset, train2db
from plot import plot_comparison_ae, save_history_graph, ae_correlation
from image_data_helpers import get_data


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
        f.write(f'Evaluation time: {(eval_time):.2f} ms\n')
        f.write(f'Scores (MSE): {scores}\n')
        f.write(f'Scores (r2): {r2}\n')
        f.write('\n***** end of file *****')


if __name__ == '__main__':
    # set metal backend (apple socs)
    device = torch.device(
        'mps' if torch.backends.mps.is_available() else 'cpu')

    name = input("Enter model name: ")
    root = Path.cwd()

    out_dir = root/'created_models'/'autoencoder'/'64x64'/name
    # out_dir = root/'created_models'/'autoencoder'/'32x32'/name
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    test_pair = (300, 60)
    val_pair = (400, 45)
    resolution = 64
    is_square = True

    train, test, val = get_data(test_pair, val_pair, resolution, square=is_square)

    dataset = TensorDataset(torch.tensor(train, device=device, dtype=torch.float32))
    trainloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # hyperparameters (class property?)
    epochs = 500
    learning_rate = 1e-3
    model = A64_7().to(device)  # move model to gpu
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    epoch_loss = []
    epoch_validation = []
    loop = tqdm(range(epochs), desc='Training...', unit='epoch', colour='#7dc4e4')

    train_start = time.time()
    for epoch in loop:
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

        if (epoch+1) % epochs == 0:
            # save model every 10 epochs (so i dont lose all training progress in case i do something unwise)
            torch.save(model.state_dict(), out_dir/f'{name}')
            save_history_graph(epoch_loss, out_dir)

    train_end = time.time()

    with torch.no_grad():
        encoded = model.encoder(torch.tensor(test, device=device, dtype=torch.float32))
        decoded = model(torch.tensor(test, device=device, dtype=torch.float32))

    torch.save(model.state_dict(), out_dir/f'{name}')
    train2db(out_dir, name, epochs, test_pair[0], test_pair[1], resolution, typ='autoencoder')
    eval_time, scores = plot_comparison_ae(test, encoded, model, out_dir=out_dir, is_square=True, resolution=resolution)
    r2 = ae_correlation(test, decoded, out_dir)
    plot_train_loss(epoch_loss, epoch_validation)
    write_metadata(out_dir)
