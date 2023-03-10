"""Autoencoder for predicting 2d plasma profiles

Jupyter eats up massive RAM so I'm making a script to do my tests
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import TensorDataset, DataLoader

# from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':
    # set metal backend (apple socs)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    device

    root = Path.cwd().parents[0]
    data = root/'data'/'interpolation_datasets'/'rec-interpolation2.nc'
    ds = xr.open_dataset(data)
    ds