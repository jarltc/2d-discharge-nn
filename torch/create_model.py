"""
Load data and train a model. (version 3)

6 Dec 13:30
author @jarl
"""

import os
# import sys
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

import data


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


# --------------- Model hyperparameters -----------------
# model name
name = input('Enter model name: ') or 'test'

root = Path.cwd().parents[0]  # the script will be in 2d-discharge-nn/torch, so move up one level
data_fldr_path = root/'data'/'avg_data'

# training inputs
batch_size = int(input('Batch size (default 128): ') or '128')
learning_rate = float(input('Learning rate (default 0.001): ') or '0.001')
validation_split = float(input('Validation split (default 0.1): ') or '0.1')
epochs = int(input('Epochs (default 5): ') or '5')
xy = data.yn(input('Include xy augmentation? [y/n] (default y): ')) or True
vp = data.yn(input('Include vp augmentation? [y/n] (default y): ')) or True
minmax_y = True  # apply minmax scaling to targets 
lin = True  # scale the targets linearly

# -------------------------------------------------------

voltages  = [200, 300, 400, 500] # V
pressures = [  5,  10,  30,  45, 60, 80, 100, 120] # Pa


voltage_excluded = 300 # V
pressure_excluded = 60 # Pa

# -------------------------------------------------------
if ((name=='test') & (root/'created_models'/'test_dir').exists()):
    out_dir = root/'created_models'/'test_dir'  
elif ((name=='test') & (not (root/'created_models'/'test_dir').exists())): 
    os.mkdir(root/'created_models'/'test_dir')
else:
    out_dir = data.create_output_dir(root) 

scaler_dir = out_dir / 'scalers'
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
labels = data.data_preproc(data_used[label_names]).astype('float64')

if minmax_y:  # if applying minmax to target data
    labels = data.scale_all(labels, 'y', scaler_dir)

alldf = pd.concat([features, labels], axis=1)  # what is this used for??
dataset_size = len(alldf)

# create dataset object and shuffle it()  # TODO: train/val split
dataset = torch.utils.data.TensorDataset(features.to_numpy(), labels.to_numpy())

# determine validation split
# trainset, valset = torch.utils.data.random_split(dataset, 
#                                                  lengths=[validation_split, (1-validation_split)],
#                                                  generator=torch.Generator().manual_seed(64))

trainloader = torch.utils.data.DataLoader(dataset, 
                                          batch_size=batch_size, 
                                          shuffle=True,
                                          num_workers=2)


# create validation split and batch the data

model = MLP(name, len(feature_names), len(label_names)) 
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train the model
print('begin model training...')
train_start = time.time()  # record start time

epoch_times = []

for epoch in tqdm(range(epochs)):  # TODO: validation data
    epoch_start = time.time()  # record time per epoch

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss/4:.3f}\r')
        running_loss = 0.0

    epoch_end = time.time()
    epoch_times.append(epoch_end - epoch_start)
        # if i % 4 == 3:  # print every 4 mini batches (?)
        #     model.eval()
        #     val_pred = model(valset)

print('Finished training')
train_end = time.time()  # record end time

# save the model
model.save(out_dir / 'model')
print('NN model has been saved.\n')

# save_history_vals(history, out_dir)
# save_history_graph(history, out_dir, 'mae')
# save_history_graph(history, out_dir, 'loss')
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

# human-readable metadata
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
    f.write('\n*** end of file ***\n')
