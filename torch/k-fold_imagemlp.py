"""
k-fold validation of the imagemlp
Created on Wed 11 Oct 15:00 2023

@author: jarl

1. split the dataset into k-groups
2. take one group as the test data, and take the remaining groups as training
3. train autoencoder
4. train imagemlp using pretrained autoencoder
5. retain the evaluation score and discard the model
"""


from pathlib import Path
import itertools
from tqdm import tqdm
import datetime as dt
import numpy as np

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
torch.manual_seed(131745)  # 911
import torch.nn as nn
import torch.optim as optim
import torchvision

from image_data_helpers import get_data, AugmentationDataset
from data_helpers import mse
from plot import ae_correlation


from autoencoder_classes import A64_8
from mlp_classes import MLP3


def train_autoencoder():
    global device
    ncfile = Path('/Users/jarl/2d-discharge-nn/data/interpolation_datasets/synthetic/synthetic_averaged.nc')
    resolution = 64

    # load data and send to device (cpu or gpu)
    augdataset = AugmentationDataset(ncfile.parent, device, resolution=resolution)
    trainloader = DataLoader(augdataset, batch_size=32, shuffle=True)

    # initialize model
    model = A64_8().to(device)

    # hyperparameters
    epochs = 500
    learning_rate = 1e-3
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loop = tqdm(range(epochs), desc='Training AE...', unit='epoch', colour='#7dc4e4')

    # model training
    for epoch in loop:
        for i, batch_data in enumerate(trainloader):
            # get inputs
            inputs = batch_data 
            optimizer.zero_grad()

            # record loss
            running_loss = 0.0

            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_description(f"Epoch {epoch+1}/{epochs}")

    # after training, return the model with its weights for use in the imageMLP
    model.eval()
    return model


def train_mlp(test_set, ae_model):
    global device

    encodedx = 40
    encodedy = encodedz = 8
    encoded_size = encodedx*encodedy*encodedz
    # ----- #
    epochs = 500
    learning_rate = 1e-3
    dropout_prob = 0.5
    optimizer = optim.Adam(mlp.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    # ----- #
    mlp = MLP3(2, encoded_size, dropout_prob=dropout_prob)  # dropout_prob doesnt do anything in MLP3

    # get data
    train, test = get_data(test=test_set, resolution=64, labeled=True)
    train_images, train_labels = train
    test_image, test_label = test

    dataset = TensorDataset(torch.tensor(train_images, device=device, dtype=torch.float32),
                            torch.tensor(train_labels, device=device, dtype=torch.float32))
    trainloader = DataLoader(dataset, batch_size=1, shuffle=True)

    loop = tqdm(range(epochs), desc='Training MLP...', unit='epoch', colour='#7dc4e4')

    print("Training MLP...\r", end="")
    for epoch in loop:
        for i, batch_data in enumerate(trainloader):
            image, labels = batch_data  # feed in images and labels (V, P)

            optimizer.zero_grad()  # reset gradients

            encoding = mlp(labels).reshape(1, encodedx, encodedy, encodedz)  # forward pass, get mlp prediction from (v, p) then reshape
            output = ae_model.decoder(encoding)  # get output image from decoder
            output = torchvision.transforms.functional.crop(output, 0, 0, 64, 64)  # crop to match shape

            loss = criterion(output, image)  # get the loss between input image and model output
            loss.backward()  # backward propagation
            optimizer.step()  # apply changes to network

            # print statistics
            loop.set_description(f"Epoch {epoch+1}/{epochs}")

    print("\33[2KMLP training complete!")
    mlp.eval()

    with torch.no_grad():
        fake_encoding = mlp(torch.tensor(test_label, device=device, dtype=torch.float32))  # mps does not support float64
        # reshape encoding from (1, xyz) to (1, x, y, z)
        fake_encoding = fake_encoding.reshape(1, encodedx, encodedy, encodedz)
        decoded = ae_model.decoder(fake_encoding).cpu().numpy()[:, :, 64, 64]

    return test_image, decoded


if __name__ == "__main__":
    voltages = [300, 400]
    pressures = [30, 45, 60, 80, 100]
    vps = [(v, p) for v in voltages for p in pressures]  # create list of v and p to leave out
    # k = 10

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    root = Path.cwd()
    out_dir = root/'kfold'
    if not out_dir.exists():
        out_dir.mkdir()

    loop = tqdm(vps, desc=f'testing v={vps[0]}, p={vps[1]}', unit='k', colour='#7dc4e4')
    today = dt.datetime.now().strftime('%Y-%m-%dd %H:%M:%S')
    filename = today + '.txt'

    with open(out_dir/filename, 'a') as file:
        file.write(f'k-fold validation for k = {len(vps)}')
        file.write('variables are (potential, electron density, ion density, \
                   metastable density, electron temperature)')
        for vp in loop:
            print(f'test case {vp}')
            file.write(f'training for excluded set {vp}:')
            
            print(f'Training autoencoder: ')
            ae_model = train_autoencoder()
            
            print(f'Training MLP: ')
            original, prediction = train_mlp(vp, ae_model)
            
            # compute scores
            rsquare = [ae_correlation(original[0, i], prediction[0, i]) for i in range(5)]
            file.write(rsquare)
            meansquareerror = [mse(original[0,i], prediction[0,i]) for i in range(5)]
            file.write(meansquareerror)
            file.write('\n')
            

