""" Testing bare training functionality for remote GPU computer """

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time

import numpy as np
from tqdm import tqdm

from autoencoder_classes import A64_9_BN
from data_helpers import set_device

if __name__ == "__main__":
    # set stuff
    dtype = torch.float32
    device = set_device()
    model = A64_9_BN()
    
    # create random data
    npdata = np.random.uniform(0, 1, size=(10, 5, 64, 64))
    train_data = torch.tensor(npdata, dtype=dtype)  
    trainloader = DataLoader(train_data, batch_size=32, shuffle=True)  

    # set hyperparameters
    epochs = 500
    learning_rate = 1e-3
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # train model for N epochs
    loop = tqdm(range(epochs), desc='Training...', unit='epoch', colour='#7dc4e4')
    for epoch in loop:
        for i, batch_data in enumerate(trainloader):
            inputs = batch_data
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()