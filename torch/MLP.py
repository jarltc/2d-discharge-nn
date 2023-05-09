"""
Load data and train a model. (PyTorch version)

15 Feb 17:25
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
import torch.multiprocessing as mp

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import data
import plot

torch.set_default_dtype(torch.float64)
class MLP(nn.Module):
    """Neural network model for grid-wise prediction of 2D-profiles.

    Model architecture optimized using OpTuna.
    Args:
        name (string): Model name
        input_size (int): Size of input vector.
        output_size (int): Size of output vector. Should be 5 for the usual variables.
    """
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
        """Execute the forward pass.

        Args:
            x (torch.Tensor): Input tensor of size (batch_size, input_size)

        Returns:
            torch.Tensor: Predicted values given an input x.
        """
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
def process_batch(chunk: torch.Tensor, model: MLP, df: pd.DataFrame, k=4) -> torch.Tensor: 
    """Batch-wise processing of the input tensor.

    Takes a batch of points as an input, along with the current model, and a DataFrame of 
    grid coordinates. The cKDTree is also created from this df.
    Args:
        chunk (torch.Tensor): _description_
        model (MLP): _description_
        df (pd.DataFrame): _description_

    Returns:
        torch.Tensor: Tensor of size (chunk_size, 5) containing neighbor means of the input chunk.
    """

    def neighbor_mean(point: torch.Tensor, k):
        # get a point's neighbors
        x, y, v, p = point.numpy() # -> np.ndarray
        v = np.atleast_1d(v)  # converts to arrays with at least one dimension
        p = np.atleast_1d(p)
        _, ii = tree.query([x, y], k, distance_upper_bound=1e-3)  # get indices of k neighbors of the point (max: 1e-3m)
        
        neighbor_xy = [df[['X', 'Y']].iloc[i].to_numpy() for i in ii]  # size: (k, 5 vars)
        neighbors = [np.concatenate((xy, v, p)) for xy in neighbor_xy]  # list of input vectors x
        neighbors = [torch.tensor(neighbor).expand(1, -1) for neighbor in neighbors]  # expand to match sizes (i forgot why)

        # concat neighbors and get the mean for each variable
        mean_tensors = torch.cat([model(neighbor) for neighbor in neighbors], dim=0)
        return torch.mean(mean_tensors, dim=0)
    
    tree = cKDTree(np.c_[df['X'].to_numpy(), df['Y'].to_numpy()])

    with torch.no_grad():
        results = [neighbor_mean(x, 4) for x in chunk]
    results = torch.stack(results, dim=0)
    return results


def worker(queue, results, model, df, k):
    # function to be executed by each process
    # runs in an infinite loop until a batch is added to the queue
    while True:
        chunk = queue.get()  # retrieve data if a batch is added
        if chunk is None:  # worker is active until a None is added to the queue
            break  # terminate the process
        output = process_batch(chunk, model, df, k)
        results.put(output)  # add output to the results queue


def c_e(epoch, c=0.2, r=25, which='sigmoid'):
    """Get regularization coefficient.

    c is generated following an exponential function

    Args:
        epoch (int): Current epoch.
        c (float, optional): Regularization coefficient to be approached. Defaults to 0.5.
        r (int, optional): Rate of increase. Coefficient increases to c/2 over r epochs. Defaults to 25.

    Returns:
        c: Coefficient at each epoch.
    """

    if which=='exp':
        return c - c*np.exp(-epoch/r)
    elif which=='sigmoid':
        return c/(1 + np.exp(-0.5*(epoch-r)))
#######################################

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
            if neighbor_regularization:
                f.write(f'Neighbor regularization: k = {k}, lambda = {c}\n')
            else:
                f.write(f'Neighbor regularization: none')
            f.write('\n*** end of file ***\n')


if __name__ == '__main__':
    # --------------- Model hyperparameters -----------------

    root = Path.cwd() 
    data_fldr_path = root/'data'/'avg_data'
    inputFile = root/'torch'/sys.argv[1] # root/'torch'/'M501.txt'

    # read input file
    with open(inputFile, 'r') as i:
        lines = i.readlines()
    
    lines = [line.split()[-1] for line in lines]

    minmax_y = True  # apply minmax scaling to targets 
    lin = True  # scale the targets linearly

    # read metadata from input file
    name = lines[0]
    batch_size = eval(lines[1])
    learning_rate = eval(lines[2])
    validation_split = eval(lines[3])
    epochs = eval(lines[4])
    xy = eval(lines[5])
    vp = eval(lines[6])
    k = eval(lines[7])  # number of neighbors, 0 to disable

    if k==0:
        neighbor_regularization = False
        c = eval(lines[8])  # neighbor regularization lambda
    else:
        c = 0

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

    alldf = pd.concat([features, labels], axis=1)  # TODO: consider removing this
    dataset_size = len(alldf)

    # kD tree for neighbor regularization
    nodes_df = data.scale_all(data_excluded[['X', 'Y']], 'x') 

    # create dataset object and shuffle it()  # TODO: train/val split
    features = torch.tensor(features.to_numpy())
    labels = torch.tensor(labels.to_numpy())
    dataset = TensorDataset(features, labels)

    trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MLP(name, len(feature_names), len(label_names)) 
    model.share_memory()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # train the model
    print('begin model training...')
    train_start = time.time()  # record start time

    epoch_times = []
    epoch_loss = []
    train_losses = []
    neighbor_losses = []

    model.train()
    # model training loop
    for epoch in tqdm(range(epochs)):
        # record time per epoch
        epoch_start = time.time()
        loop = tqdm(trainloader)

        for i, batch_data in enumerate(loop):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = batch_data

            if neighbor_regularization:
                # means not calculated if regularization is disabled
                c = c_e(epoch, c=c)
                neighbor_means = process_batch(inputs, model, nodes_df) 
            else:
                neighbor_means = 0
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # record losses
            running_loss = 0.0

            outputs = model(inputs)  # forward pass
            neighbor_loss = criterion(outputs, neighbor_means)
            train_loss = criterion(outputs, labels)
            loss = train_loss + c*neighbor_loss  # second term 0 if neighbor_regularization turned off
            loss.backward()  # compute gradients
            optimizer.step()  # apply changes to network

            # print statistics
            running_loss += loss.item()
            loop.set_description(f"Epoch {epoch+1}/{epochs}")
            loop.set_postfix(loss=running_loss)
            running_loss = 0.0

        epoch_end = time.time()
        epoch_times.append(epoch_end - epoch_start)
        epoch_loss.append(loss.item())
        train_losses.append(train_loss.item())
        neighbor_losses.append(neighbor_loss.item())

        if (epoch+1) % epochs == 0:
            # save model every 10 epochs (so i dont lose all training progress in case i do something dumb)
            torch.save(model.state_dict(), out_dir/f'{name}')
            plot.save_history_graph(epoch_loss, out_dir)


    print('Finished training')
    train_end = time.time()

    # save the model and loss
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
