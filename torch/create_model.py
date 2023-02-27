"""
Load data and train a model. (version 3)

6 Dec 13:30
author @jarl
"""

import os
import sys
import time
import datetime
import shutil
import pickle
from tqdm import tqdm
from pathlib import Path

import numpy as np
import pandas as pd
# import xarray as xr
# import matplotlib
# import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import data
import plot


class MLP(nn.Module):
    def __init__(self, name, input_size, output_size) -> None:
        super(MLP, self).__init__()
        self.name = name
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, 115)  # linear: y = Ax + b
        self.fc2 = nn.Linear(115, 78)
        self.fc3 = nn.Linear(78, 26)
        self.fc4 = nn.Linear(26, 46)
        self.fc5 = nn.Linear(46, 82)
        self.fc6 = nn.Linear(82, 106)
        self.fc7 = nn.Linear(106, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)
        x = self.fc6(x)
        x = F.relu(x)
        x = self.fc7(x)
        
        output = x = F.relu(x)
        return output


####### neighbor regularization #######
def neighbor_mean(point, k):
    global tree, model
    model.eval()

    # get a point's neighbors and get a vector of the means for each variable (shape: (k,5))
    x, y, v, p = point.numpy() # -> np.ndarray
    v = v.expand_dims(axis=0)
    p = v.expand_dims(axis=0)

    _, ii = tree.query([x, y], k)  # get indices of k neighbors of the point
    
    # get pair of x, y of point's neighbors:
    # 1. combine with v, p as input to the model
    # 2. convert to tensor (add a new dim cause the model expects a batch size)
    # 3. get mean of neighbor predictions for 5 variables

    neighbor_xy = [features[['x', 'y']].iloc[i].to_numpy() for i in ii]  # size: (k, 5)
    neighbors = [np.concatenate((xy, v, p)) for xy in neighbor_xy]  # list of input vectors x
    neighbors = [torch.tensor(neighbor).expand(1, -1, -1) for neighbor in neighbors]

    # dim 0 gets the mean betw rows
    return torch.mean([model(neighbor) for neighbor in neighbors], dim=0)


def neighbor_loss(x_batch: torch.Tensor, y_batch: torch.Tensor, training=False, k=3):
    """Calculate neighbor difference for each training point in a batch.

    This loss is added as a regularizer to improve smoothness.
    Args:
        x_batch (torch.Tensor): Batched features of the specified batch size.
        y_batch (torch.Tensor): Batched labels of the specified batch size.
        training (bool, optional): Make model weights trainable. Defaults to False.
        k (int, optional): Number of neighbors to query. Defaults to 4.

    Returns:
        Total MSE: torch.Tensor of shape(batch_size, ) specifying the MSE for each item in the batch.
    """
    c = 0.3  # parameter for the neighbor loss

    # neighbor_mean returns a tensor of neighbor means for each point in the batch
    y_pred = model(x_batch, training=training)
    neighbor_means = [neighbor_mean(point, k) for point in x_batch]    
    batch_mean = torch.cat(neighbor_means, dim=0)

    def neighbor_loss_core(y_true, y_pred):
        """Actual loss function.

        Args:
            y_true (tf.Tensor): Tensor of shape (batch_size, num_labels).
            y_pred (tf.Tensor): Output of model(x_batch), with a shape of (batch_size, num_labels)

        Returns:
            Total MSE: tf.Tensor of shape(batch_size, ) specifying the MSE for each item in the batch.
        """
        mse_train = nn.MSELoss(y_pred, y_true)
        mse_neighbor = c*nn.MSELoss(batch_mean, y_pred)

        return torch.add(mse_train, mse_neighbor)

    return neighbor_loss_core(y_pred, y_batch)


def save_history_vals(history, out_dir):
    """Save history values

    Args:
        history (_type_): _description_
        out_dir (_type_): _description_
    """
    history_path = out_dir / 'history.csv'
    history_table = np.hstack([np.array(history.epoch              ).reshape(-1,1),
                               np.array(history.history['mae']     ).reshape(-1,1),
                               np.array(history.history['val_mae'] ).reshape(-1,1),
                               np.array(history.history['loss']    ).reshape(-1,1),
                               np.array(history.history['val_loss']).reshape(-1,1)])
    history_df = pd.DataFrame(history_table, columns=['epoch','mae','val_mae','loss','val_loss'])
    history_df.to_csv(history_path, index=False)


def save_metadata(out_dir: Path):
    """Save readable metadata.

    Args:
        out_dir (Path): Path to where the model is saved.
    """
    with open(out_dir / 'train_metadata.txt', 'w') as f:
            f.write(f'Model name: {name}\n')
            f.write(f'Lin scaling: {lin}\n')
            f.write(f'Number of points: {len(data_used)}\n')
            f.write(f'Target scaling: {minmax_y}\n')
            f.write(f'Parameter exponents: {scale_exp}\n')
            f.write(f'Execution time: {(train_end-train_start):.2f} s\n')
            f.write(f'Average time per epoch: {np.array(epoch_times).mean():.2f} s\n')
            f.write(f'\nUser-specified hyperparameters\n')
            f.write(f'Batch size: {batch_size}\n')
            f.write(f'Learning rate: {learning_rate}\n')
            f.write(f'Validation split: {validation_split}\n')
            f.write(f'Epochs: {epochs}\n')
            f.write(f'Grid augmentation: {xy}\n')
            f.write(f'VP augmentation: {vp}\n')
            f.write('\n*** end of file ***\n')


if __name__ == '__main__':
    # --------------- Model hyperparameters -----------------
    # name = input('Enter model name: ') or 'test'
    name = 'test'

    root = Path.cwd() 
    data_fldr_path = root/'data'/'avg_data'

    # training inputs
    # batch_size = int(input('Batch size (default 128): ') or '128')
    # learning_rate = float(input('Learning rate (default 0.001): ') or '0.001')
    # validation_split = float(input('Validation split (default 0.1): ') or '0.1')
    # epochs = int(input('Epochs (default 5): ') or '5')
    # xy = data.yn(input('Include xy augmentation? [y/n]: ')) 
    # vp = data.yn(input('Include vp augmentation? [y/n]: '))

    # testing
    batch_size = 128
    learning_rate = 0.001
    validation_split = 0.1
    epochs = 3
    xy = False
    vp = False
    minmax_y = True  # apply minmax scaling to targets 
    lin = True  # scale the targets linearly

    # -------------------------------------------------------

    voltages  = [200, 300, 400, 500] # V
    pressures = [  5,  10,  30,  45, 60, 80, 100, 120] # Pa


    voltage_excluded = 300 # V
    pressure_excluded = 60 # Pa

    # -------------------------------------------------------

    if ((name=='test') & (root/'created_models'/'test_dir_torch').exists()):
        out_dir = root/'created_models'/'test_dir_torch'  
    elif ((name=='test') & (not (root/'created_models'/'test_dir_torch').exists())): 
        os.mkdir(root/'created_models'/'test_dir_torch')
    else:
        out_dir = data.create_output_dir(root) 

    scaler_dir = out_dir / 'scalers'
    if (not scaler_dir.exists()):
        os.mkdir(scaler_dir) 

    # copy some files for backup (probably made redundant by metadata)
    shutil.copyfile(__file__, out_dir / 'create_model.py')
    shutil.copyfile(root / 'data.py', out_dir / 'data.py')

    feature_names = ['V', 'P', 'x', 'y']
    label_names = ['potential (V)', 'Ne (#/m^-3)', 'Ar+ (#/m^-3)', 'Nm (#/m^-3)', 'Te (eV)']

    data_used, data_excluded = data.get_data(root, voltages, pressures, 
                                            (voltage_excluded, pressure_excluded),
                                            xy=xy, vp=vp)
    # sanity check
    assert list(data_used.columns) == feature_names + label_names

    # set threshold to make very small values zero
    pd.set_option('display.chop_threshold', 1e-10)

    # scale features and labels
    scale_exp = []
    features = data.scale_all(data_used[feature_names], 'x', scaler_dir).astype('float64')
    labels = data.data_preproc(data_used[label_names], scale_exp).astype('float64')

    if minmax_y:  # if applying minmax to target data
        labels = data.scale_all(labels, 'y', scaler_dir)

    alldf = pd.concat([features, labels], axis=1)  # TODO: think about removing this
    dataset_size = len(alldf)

    # kD tree for neighbor regularization
    nodes_df = scale_all(data_excluded[['X', 'Y']], 'x') 
    tree = cKDTree(np.c_[nodes_df['X'].to_numpy(), nodes_df['Y'].to_numpy()])

    # create dataset object and shuffle it()  # TODO: train/val split
    features = torch.FloatTensor(features.to_numpy())
    labels = torch.FloatTensor(labels.to_numpy())
    dataset = TensorDataset(features, labels)

    trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MLP(name, len(feature_names), len(label_names)) 
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # train the model
    print('begin model training...')
    train_start = time.time()  # record start time

    epoch_times = []
    epoch_loss = []

    for epoch in tqdm(range(epochs)):  # TODO: validation data
        epoch_start = time.time()  # record time per epoch
        loop = tqdm(trainloader)

        # record losses
        running_loss = 0.0

        for i, batch_data in enumerate(loop):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = batch_data

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(inputs)  # forward pass
            loss = criterion(outputs, labels)
            loss.backward()  # compute gradients
            optimizer.step()  # apply changes to network

            # print statistics
            running_loss += loss.item()
            loop.set_description(f"Epoch {epoch}/{epochs}")
            # loop.set_postfix(loss=running_loss, epoch_loss=epoch_loss)
            running_loss = 0.0

        epoch_end = time.time()
        epoch_times.append(epoch_end - epoch_start)
        epoch_loss.append(loss.item())

    print('Finished training')
    train_end = time.time()  # record end time

    # save the model
    torch.save(model.state_dict(), out_dir/f'{name}')
    print('NN model has been saved.\n')
    
    plot.save_history_graph(epoch_loss, out_dir)
    print('NN training history has been saved.\n')

    # save metadata
    metadata = {'name' : name,  # str
                'scaling' : lin,  # bool
                'is_target_scaled': minmax_y,  # bool
                'parameter_exponents': scale_exp}  # list of float

    with open(out_dir / 'train_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

    # record time per epoch
    with open(out_dir / 'times.txt', 'w') as f:
        f.write('Train times per epoch\n')
        for i, time in enumerate(epoch_times):
            time = round(time, 2)
            f.write(f'Epoch {i+1}: {time} s\n')

    d = datetime.datetime.today()
    print('finished on', d.strftime('%Y-%m-%d %H:%M:%S'))

    save_metadata(out_dir)
