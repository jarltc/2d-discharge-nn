"""Autoencoder for predicting 2d plasma profiles

Jupyter eats up massive RAM so I'm making a script to do my tests
"""

import os
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
from torch.utils.data import TensorDataset, DataLoader

# from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from plot import draw_apparatus

# define model TODO: construct following input file/specification list
class Autoencoder(nn.Module):
  def __init__(self):
    super(Autoencoder, self).__init__()
    self.encoder = nn.Sequential(
      nn.Conv2d(5, 16, kernel_size=5, stride=2, padding=1),
      nn.ReLU(),
      nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1),
      nn.ReLU(),
    )

    self.decoder = nn.Sequential(
      nn.ConvTranspose2d(8, 16, 3, 2),
      nn.ConvTranspose2d(16, 8, 3, 2),
      nn.Conv2d(8, 5, 3, padding=1)
    )


  def forward(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    decoded = torchvision.transforms.functional.crop(decoded, 0, 0, 707, 200) 
    return decoded


class Trial(): # TODO: add plotting
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


def nc_to_tensor(ds):
  variables = list(ds.data_vars)
  v_excluded = 300  # TODO: move somewhere else
  p_excluded = 60

  data_list = []
  excluded_list = []
  for v in ds.V.values:
    for p in ds.P.values:
        vp_data = np.nan_to_num(np.stack([scale_np(ds[var].sel(V=v, P=p).values, var, scaler_dict) for var in variables]))
        if (v == v_excluded) & (p == p_excluded):
          excluded_list.append(vp_data)
        else:
          data_list.append(vp_data)

  train = np.float32(np.stack(data_list))  # consider saving as .pt file after conversion
  test = np.float32(np.stack(excluded_list))
   
  # # shuffle dimensions and combine v, p
  # data = np.moveaxis(data, 0, 4)
  # data = data.reshape(32,707,200,5)
  # data = np.moveaxis(data, 3, 1)
  assert train.shape == (31, 5, 707, 200)  # samples, channels, height, width

  return train, test


def scale_np(array, var, scaler_dict):
  max = np.nanmax(array)
  min = np.nanmin(array) 

  scaler_dict[var] = (min, max)

  return (array - min) / (max - min)


def plot_comparison(reference: np.ndarray, name=None, out_dir=None):
    fig = plt.figure(figsize=(12, 7), dpi=200)
    subfigs = fig.subfigures(nrows=2)

    axs1 = subfigs[0].subplots(nrows=1, ncols=5)
    axs2 = subfigs[1].subplots(nrows=1, ncols=5)

    subfigs[1].suptitle('Reconstruction')
    subfigs[0].suptitle('Original')

    cbar_ranges = [(reference[0,i,:,:].min(), reference[0,i,:,:].max()) for i in range(5)]

    with torch.no_grad():
        encoded = model.encoder(torch.tensor(reference, device=device))
        reconstruction = model.decoder(encoded).cpu().numpy()
    
    for i in range(5):
        org = axs1[i].imshow(reference[0,i,:,:], origin='lower', extent=[0,20,0, 70.7], cmap='Greys_r')
        draw_apparatus(axs1[i])
        plt.colorbar(org)
        rec = axs2[i].imshow(reconstruction[0,i,:,:], origin='lower', extent=[0,20,0, 70.7],
                            vmin=cbar_ranges[i][0], vmax=cbar_ranges[i][1], cmap='Greys_r')
        draw_apparatus(axs2[i])
        plt.colorbar(rec)

    if out_dir is not None:
       fig.savefig(out_dir/f'test_comparison.png')


def plot_train_loss(losses):
    losses = np.array(losses)
    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.ticklabel_format(axis='y',style='sci')

    fig.savefig(out_dir/'train_loss.png')


def write_metadata(out_dir):
   # save model structure
    file = out_dir/'train_log.txt'
    with open(file, 'w') as f:
       f.write(f'Model {name}\n')
       print(model, file=f)
       f.write(f'\nEpochs: {epochs}\n')
       f.write(f'Learning rate: {learning_rate}\n')
       f.write('***** end of file *****')

if __name__ == '__main__':
    # set metal backend (apple socs)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    name = input("Enter model name: ")
    root = Path.cwd()
    data = root/'data'/'interpolation_datasets'/'rec-interpolation2.nc'
    out_dir = root/'created_models'/'autoencoder'/name
    if not out_dir.exists():
       out_dir.mkdir(parents=True)
    ds = xr.open_dataset(data)

    scaler_dict = {}
    train, test = nc_to_tensor(ds)

    dataset = TensorDataset(torch.tensor(train, device=device))
    trainloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # hyperparameters (class property?)
    epochs = 300
    learning_rate = 1e-3

    model = Autoencoder()
    model.to(device)  # move model to gpu
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # include as class property?
    weights_np = np.expand_dims(np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32), axis=(0, 2, 3))
    weights = torch.tensor(weights_np, device=device)

    # convert to class method?
    epoch_loss = []
    loop = tqdm(range(epochs))
    for epoch in loop:
        # loop = tqdm(trainloader)
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
    
    plot_comparison(test, out_dir=out_dir)
    plot_train_loss(epoch_loss)
    write_metadata(out_dir)
