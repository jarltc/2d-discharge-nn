"""
Load data and train a model. (PyTorch)

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


class MLP(nn.Module):
    """Neural network momdel for grid-wise prediction of 2D-profiles.

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
def process_chunk(chunk: torch.Tensor, model: MLP, df: pd.DataFrame) -> torch.Tensor: 
    """Per-chunk processing of the input tensor.

    Takes a chunk of a batch as an input, along with the current model, and a DataFrame of 
    grid coordinates. The cKDTree is also created from this df.
    Args:
        chunk (torch.Tensor): _description_
        model (MLP): _description_
        df (pd.DataFrame): _description_

    Returns:
        torch.Tensor: Tensor of size (chunk_size, 5) containing neighbor means of the input chunk.
    """

    def neighbor_mean(point: torch.Tensor, k: int):
        # get a point's neighbors
        x, y, v, p = point.numpy() # -> np.ndarray
        v = np.atleast_1d(v)
        p = np.atleast_1d(p)
        _, ii = tree.query([x, y], k)  # get indices of k neighbors of the point
        
        neighbor_xy = [df[['X', 'Y']].iloc[i].to_numpy() for i in ii]  # size: (k, 5)
        neighbors = [np.concatenate((xy, v, p)) for xy in neighbor_xy]  # list of input vectors x
        neighbors = [torch.FloatTensor(neighbor).expand(1, -1) for neighbor in neighbors]

        # concat neighbors and get the mean for each variable
        mean_tensors = torch.cat([model(neighbor) for neighbor in neighbors], dim=0)
        return torch.mean(mean_tensors, dim=0)
    
    tree = cKDTree(np.c_[df['X'].to_numpy(), df['Y'].to_numpy()])

    with torch.no_grad():
        results = [neighbor_mean(x, 4) for x in chunk]
    results = torch.stack(results, dim=0)
    return results


def worker(queue, results, model, df):
    # function to be executed by each process
    # runs in an infinite loop until a batch is added to the queue
    while True:
        chunk = queue.get()  # retrieve data if a batch is added
        if chunk is None:  # worker is active until a None is added to the queue
            break  # terminate the process
        output = process_chunk(chunk, model, df)
        results.put(output)  # add output to the results queue


def c_e(epoch, c=0.5, r=25, which='exp'):
    """Get regularization coefficient.

    c is generated following an exponential function

    Args:
        epoch (int): Current epoch.
        c (float, optional): Regularization coefficient to be approached. Defaults to 0.5.
        r (int, optional): Rate of increase. Coefficient increases by {} over r epochs. Defaults to 25.

    Returns:
        c: Coefficient at each epoch.
    """

    if which=='exp':
        return c - c*np.exp(-epoch/r)
    elif which=='sigmoid':
        k = 0.085
        x_0 = 100
        return c/(1 + np.exp(-k*(epoch-x_0)))
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

    name = lines[0]
    batch_size = eval(lines[1])
    learning_rate = eval(lines[2])
    validation_split = eval(lines[3])
    epochs = eval(lines[4])
    xy = eval(lines[5])
    vp = eval(lines[6])
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

    alldf = pd.concat([features, labels], axis=1)  # TODO: consider removing this
    dataset_size = len(alldf)

    # kD tree for neighbor regularization
    nodes_df = data.scale_all(data_excluded[['X', 'Y']], 'x') 

    # create dataset object and shuffle it()  # TODO: train/val split
    features = torch.FloatTensor(features.to_numpy())
    labels = torch.FloatTensor(labels.to_numpy())
    dataset = TensorDataset(features, labels)

    trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MLP(name, len(feature_names), len(label_names)) 
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # initialize multiprocessing for neighbor mean
    num_processes = mp.cpu_count()
    chunk_size = int(batch_size/num_processes)
    queue = mp.Queue()
    results = mp.Queue()
    processes = [mp.Process(target=worker, args=(queue, results, model, nodes_df)) for _ in range(num_processes)]
    
    # signal each worker to start if not already started
    print(f'Multicore neighbor mean processing ({num_processes} cores):')
    for i, p in enumerate(processes):
        if not p.is_alive():
            p.start()
            print(f'spawned process {i+1}/{num_processes}')

    # train the model
    print('begin model training...')
    train_start = time.time()  # record start time

    epoch_times = []
    epoch_loss = []

    # model training loop
    for epoch in tqdm(range(epochs)):
        # record time per epoch
        epoch_start = time.time()
        loop = tqdm(trainloader)

        for i, batch_data in enumerate(loop):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = batch_data
            doubleLoader = DataLoader(inputs, batch_size=chunk_size)
            c = c_e(epoch)
            
            model.eval()  # switch model to eval mode

            # load data into queue
            for chunk in doubleLoader:
                queue.put(chunk)

            # retrieve processed data from the results queue
            worker_outputs = [results.get() for _ in range(num_processes)]
            # for _ in range(num_processes):
            #     output = results.get()
            #     worker_outputs.append(output)

            neighbor_means = torch.cat(worker_outputs, dim=0)

            model.train()  # put model back in train mode (?)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # record losses
            running_loss = 0.0

            outputs = model(inputs)  # forward pass
            loss = criterion(outputs, labels) + c*criterion(outputs, neighbor_means)
            loss.backward()  # compute gradients
            optimizer.step()  # apply changes to network

            # print statistics
            running_loss += loss.item()
            loop.set_description(f"Epoch {epoch}/{epochs}")
            loop.set_postfix(loss=running_loss)
            running_loss = 0.0

        epoch_end = time.time()
        epoch_times.append(epoch_end - epoch_start)
        epoch_loss.append(loss.item())

    # when finished, add a sentinel value to the queue
    for _ in range(num_processes):
        queue.put(None)

    # wait for all processes to terminate
    for p in processes:
        p.join()

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
