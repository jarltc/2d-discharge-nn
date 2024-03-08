"""
Conditional autoencoder to reproduce 2d plasma profiles from a pair of V, P

* Train autoencoder
* Take encoder and encode labeled profiles (square images)
* 
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

from data_helpers import ImageDataset, train2db, set_device
from plot import plot_comparison_ae, save_history_graph, ae_correlation
import autoencoder_classes
import mlp_classes
from image_data_helpers import get_data


def resize(data: np.ndarray, scale=64) -> np.ndarray:
    """Resize square images to 64x64 resolution by downscaling.

    Args:
        data (np.ndarray): Input data.

    Returns:
        np.ndarray: Downscaled input data.
    """

    data = np.stack([cv2.resize((np.moveaxis(image, 0, -1)), (scale, scale)) for image in data]) 
    return np.moveaxis(data, -1, 1)  # revert initial moveaxis


def normalize_test(dataset:np.ndarray, scalers:dict):
    normalized_variables = []

    for i, var in enumerate(['pot', 'ne', 'ni', 'nm', 'te']):
        x = dataset[:, i, :, :]
        xMin, xMax = scalers[var]
        scaledx = (x-xMin) / (xMax-xMin)  # shape: (31, x, x)
        normalized_variables.append(scaledx)
    
    # shape: (5, 31, x, x)
    normalized_dataset = np.moveaxis(np.stack(normalized_variables), 0, 1)  # shape: (31, 5, x, x)
    return normalized_dataset


def write_metadata_ae(out_dir):  # TODO: move to data module
    # if is_square:
    #     in_size = (1, 5, 200, 200)
    # else:
    #     in_size = (1, 5, 707, 200)

    in_size = batch_data[0].size()  # testing

    # save model structure
    file = out_dir/'train_log.txt'
    with open(file, 'w') as f:
        if message is not None:
            f.write(f'Message: {message}\n')
        f.write(f'Model {name}\n')
        print(summary(ae_model, input_size=in_size), file=f)
        print("\n", file=f)
        print(ae_model, file=f)
        print(summary(mlp, input_size=(1, 2)), file=f)
        print("\n")
        f.write(f'\nEpochs: {epochs}\n')
        f.write(f'Learning rate: {learning_rate}\n')
        f.write(f'Resolution: {resolution}\n')
        f.write(f'Train time: {(train_end-train_start):.2f} s\n')
        # f.write(
        #     f'Average time per epoch: {np.array(epoch_times).mean():.2f} s\n')
        f.write(f'Evaluation time: {(eval_time/1e6):.2f} ms\n')
        f.write(f'Scores (MSE): {scores}\n')
        f.write('\n***** end of file *****')


if __name__ == '__main__':
    # set metal backend (apple socs)
    device = set_device()
    
    name = input("Model name: ") or "CAE_test"
    message = input("Leave a note?: ") or None
    root = Path.cwd()

    # ----- #
    resolution = 64
    if resolution == 32:
        ae_model = autoencoder_classes.A300()
        encodedx = 20
        encodedy = encodedz = 4
    elif resolution == 64:
        ae_model = autoencoder_classes.A64_8()
        encodedx = 40
        encodedy = encodedz = 8
    
    encoded_size = encodedx*encodedy*encodedz

    ##### things to change #####
    model_dir = root/'created_models'/'autoencoder'/'64x64'/'A64-8'/'A64-8'  # path to autoencoder model
    
    # ----- #
    epochs = 500
    learning_rate = 1e-3
    dropout_prob = 0.5
    # ----- #
    mlp = mlp_classes.MLP4(2, encoded_size, dropout_prob=dropout_prob)
    mlp.to(device)
    
    train, test = get_data((300, 60), resolution=resolution, labeled=True)
    train_images, train_labels = train
    test_image, test_label = test

    out_dir = root/'created_models'/'conditional_autoencoder'/name
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    dataset = TensorDataset(torch.tensor(train_images, device=device, dtype=torch.float32),
                            torch.tensor(train_labels, device=device, dtype=torch.float32))
    trainloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # load autoencoder model
    ae_model.load_state_dict(torch.load(model_dir))
    ae_model.to(device)
    ae_model.eval()  # inference mode, disables training
    
    #### train MLP ####
    optimizer = optim.Adam(mlp.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    epoch_loss = []

    # begin training MLP
    print("Training MLP...\r", end="")
    train_start = time.time()
    loop = tqdm(range(epochs), desc='Training...', unit='epoch', colour='#7dc4e4')

    for epoch in loop:
        running_loss = 0.0  # record losses

        for i, batch_data in enumerate(trainloader):
            image, labels = batch_data  # feed in images and labels (V, P)

            optimizer.zero_grad()  # reset gradients

            encoding = mlp(labels).reshape(1, encodedx, encodedy, encodedz)  # forward pass, get mlp prediction from (v, p) then reshape
            output = ae_model.decoder(encoding)  # get output image from decoder
            output = torchvision.transforms.functional.crop(output, 0, 0, resolution, resolution)  # crop to match shape

            loss = criterion(output, image)  # get the loss between input image and model output
            loss.backward()  # backward propagation
            optimizer.step()  # apply changes to network

            # print statistics
            loop.set_description(f"Epoch {epoch+1}/{epochs}")
            running_loss += loss.item()

        epoch_loss.append(loss.item())
        if (epoch+1) % 10 == 0:
            # save model every 10 epochs (so i dont lose all training progress in case i do something unwise)
            torch.save(ae_model.state_dict(), out_dir/f'{name}')

    print("\33[2KMLP training complete!")
    train_end = time.time()
    torch.save(mlp.state_dict(), out_dir/f'{name}')
    save_history_graph(epoch_loss, out_dir)

    mlp.eval()

    # construct a fake encoding from a pair of (V, P) and reshape to the desired dimensions
    with torch.no_grad():
        fake_encoding = mlp(torch.tensor(test_label, device=device, dtype=torch.float32))  # mps does not support float64
        # reshape encoding from (1, xyz) to (1, x, y, z)
        fake_encoding = fake_encoding.reshape(1, encodedx, encodedy, encodedz)
        decoded = ae_model.decoder(fake_encoding)
        decoded = torchvision.transforms.functional.crop(decoded, 0, 0, resolution, resolution)

    # add resolution=64 for larger images
    eval_time, scores = plot_comparison_ae(test_image, fake_encoding, ae_model, 
                                           out_dir=out_dir, is_square=True)
    write_metadata_ae(out_dir)
    ae_correlation(test_image, decoded, out_dir, minmax=False)
    # train2db(out_dir, name, epochs, image_ds.v_excluded, image_ds.p_excluded, resolution, typ='mlp')
