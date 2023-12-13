"""Autoencoder for predicting 2d plasma profiles

This uses the full rectangular simulation image.
"""

import os
import time
import pickle
from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary

import autoencoder_classes as AE
from plot import plot_comparison_ae, save_history_graph, ae_correlation
from image_data_helpers import get_data, AugmentationDataset


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
    # in_size = batch_data[0].size()  # broken
    if resolution is not None:
        in_size = (1, 5, resolution, resolution)
    else:
        in_size = (1, 5, 707, 200)

    # save model structure
    file = out_dir/'train_log.txt'
    with open(file, 'w') as f:
        f.write(f'Model {name}\n')
        f.write('***** layer behavior *****\n')
        print(summary(model, input_size=in_size, device='mps'), file=f)
        print("\n", file=f)
        f.write('***** autoencoder architecture *****\n')
        print(model, file=f)
        f.write(f'\nEpochs: {epochs}\n')
        f.write(f'Learning rate: {learning_rate}\n')
        f.write(f'Resolution: {resolution}\n')
        f.write(f'Train time: {(train_end-train_start):.2f} seconds ({(train_end-train_start)/60:.2f} minutes)\n')
        f.write('\n***** end of file *****')

        

if __name__ == '__main__':
    # set metal backend (apple socs)
    resolution = None
    device = torch.device(
        'mps' if torch.backends.mps.is_available() else 'cpu')

    # name = input("Enter model name: ")
    name = 'fullAE-1'
    root = Path.cwd()

    out_dir = root/'created_models'/'autoencoder'/name
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    test_pair = (300, 60)
    val_pair = (400, 45)
    is_square = True
    dtype = torch.float32

    # get augmentation data
    ncfile = Path('/Users/jarl/2d-discharge-nn/data/interpolation_datasets/synthetic/synthetic_averaged.nc')

    _, test, val = get_data(test_pair, val_pair, square=False)

    augdataset = AugmentationDataset(ncfile.parent, device)
    trainloader = DataLoader(augdataset, batch_size=32, shuffle=True)
    val_tensor = torch.tensor(val, device=device, dtype=dtype)

    epochs = 100
    learning_rate = 1e-3
    model = AE.FullAE1().to(device)
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    weight_tensor = torch.tensor([1, 1, 1, 10, 1], dtype=dtype, device=device).view(1, -1, 1, 1)

    epoch_loss = []
    epoch_validation = []
    loop = tqdm(range(epochs), desc='Training...', unit='epoch', colour='#7dc4e4')

    patience = 30
    best_loss = 100
    best_epoch = -1
    eps = 1e-5  # threshold to consider if the change is significant
    epochs_since_best = 0

    train_start = time.perf_counter()
    for epoch in loop:
        for i, batch_data in enumerate(trainloader):
            # get inputs
            inputs = batch_data * weight_tensor
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
            val_loss = criterion(model(val_tensor), val_tensor).item()

        epoch_validation.append(val_loss)
        epoch_loss.append(running_loss)

        # -----  EARLY STOPPPING ----- #
        # check if the current loss is smaller than the best loss 
        # if (val_loss < best_loss):
        #     # update the best epoch and loss
        #     best_epoch = epoch  

        #     if (abs(val_loss - best_loss) < eps):
        #         # if change is insignificant, keep waiting
        #         epochs_since_best += 1

        #         if epochs_since_best >= patience:
        #             # save model when best result is reached and model has not improved a number of times
        #             epochs = best_epoch + 1
        #             torch.save(model.state_dict(), out_dir/f'{name}')
        #             save_history_graph(epoch_loss, out_dir)
        #             break
        #     else:
        #         # if loss significantly decreases, reset progress
        #         epochs_since_best = 0
            
        #     best_loss = val_loss  # update the loss after comparisons have been made


        if (epoch+1) % epochs == 0:
            # save model every 10 epochs (so i dont lose all training progress in case i do something unwise)
            torch.save(model.state_dict(), out_dir/f'{name}') 

    train_end = time.perf_counter()

    with torch.no_grad():
        encoded = model.encoder(torch.tensor(test, device=device, dtype=dtype))
        decoded = model(torch.tensor(test, device=device, dtype=dtype))

    torch.save(model.state_dict(), out_dir/f'{name}')
    # train2db(out_dir, name, epochs, test_pair[0], test_pair[1], resolution, typ='autoencoder')
    eval_time, scores = plot_comparison_ae(test, encoded, model, out_dir=out_dir, is_square=False)
    r2 = ae_correlation(test, decoded, out_dir)
    plot_train_loss(epoch_loss, epoch_validation)
    write_metadata(out_dir)
