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
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

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


def create_output_dir():
    # TODO: move to data module
    rslt_dir = root / 'created_models'
    if not os.path.exists(rslt_dir):
        os.mkdir(rslt_dir)
    
    date_str = datetime.datetime.today().strftime('%Y-%m-%d_%H%M')
    out_dir = rslt_dir / date_str
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    print('directory', out_dir, 'has been created.\n')
    
    return out_dir


def data_preproc(data_table, lin=True):
    # TODO: move to data module
    global scale_exp
    trgt_params = ('potential (V)', 'Ne (#/m^-3)', 'Ar+ (#/m^-3)', 'Nm (#/m^-3)', 'Te (eV)')

    def get_param_exp(col_vals):
        ''' get exponent of the parameter's mean value for scaling. '''
        mean_exp = round(np.log10(col_vals.mean()), 0) - 1.0
        # return 0 if mean_exp is less than zero to avoid blowing up small values
        if mean_exp >=  0.0:
            return mean_exp
        else:
            return 0.0


    for col_n,(col_name,col_vals) in enumerate(data_table.iteritems(), start=1):
        if col_name in trgt_params:
            if lin:
                # get exponent for scaling
                exponent = get_param_exp(col_vals)
                scale_exp.append(exponent)            
                tmp_col = col_vals.values.reshape(-1,1)/(10**exponent)  # scale by dividing
            else:
                tmp_col = np.log10(col_vals.values.reshape(-1,1))
        else:
            tmp_col = col_vals.values.reshape(-1,1)
        proced_table = tmp_col if col_n==1 else np.hstack([proced_table,tmp_col])
    
    proced_table = pd.DataFrame(proced_table, columns=data_table.columns)
    proced_table = proced_table.replace([np.inf,-np.inf], np.nan)
    proced_table = proced_table.dropna(how='any')
    
    return proced_table


def scale_all(data_table, x_or_y, out_dir=None):
    # TODO: move to data module
    data_table = data_table.copy()
    for n,column in enumerate(data_table.columns, start=1):
        scaler = MinMaxScaler()
        data_col = data_table[column].values.reshape(-1,1)
        scaler.fit(data_col)
        scaled_data = scaler.transform(data_col)
        scaled_data_table = scaled_data if n==1 else np.hstack([scaled_data_table,scaled_data])
        
        if out_dir is not None:
            pickle_file = out_dir / f'{x_or_y}scaler_{n:02d}.pkl'
            with open(pickle_file, mode='wb') as pf:
                pickle.dump(scaler, pf, protocol=4)
    
    scaled_data_table = pd.DataFrame(scaled_data_table, columns=data_table.columns)
    
    return scaled_data_table


def save_history_graph(history, out_dir, param='mae'):
    # TODO: move to plot module
    matplotlib.rcParams['font.family'] = 'Arial'
    x  = np.array(history.epoch)
    if param=='mae':
        y1 = np.array(history.history['mae'])
        y2 = np.array(history.history['val_mae'])
    elif param=='loss':
        y1 = np.array(history.history['loss'])
        y2 = np.array(history.history['val_loss'])
    

    plt.rcParams['font.size'] = 12 # 12 is the default size
    plt.rcParams['xtick.minor.size'] = 2
    plt.rcParams['ytick.minor.size'] = 2
    fig = plt.figure(figsize=(6.0,6.0))
    
    # axis labels
    plt.xlabel('Epoch')
    if param=='mae':
        plt.ylabel('MAE')
    elif param=='loss':
        plt.ylabel('Loss')
    
    plt.plot(x, y1, color='green', lw=1.0, label='train')
    plt.plot(x, y2, color='red'  , lw=1.0, label='test')
    
    # set both x_min and y_min as zero
    ax = plt.gca()
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    
    ax.xaxis.get_ticklocs(minor=True)
    ax.yaxis.get_ticklocs(minor=True)
    ax.minorticks_on()
    
    plt.legend()
    plt.grid()
    plt.tight_layout()
    
    # save the figure
    if param=='mae':
        file_name = 'history_graph_mae.png'
    elif param=='loss':
        file_name = 'history_graph_loss.png'
    file_path = out_dir / file_name
    fig.savefig(file_path)


def save_history_vals(history, out_dir):
    history_path = out_dir / 'history.csv'
    history_table = np.hstack([np.array(history.epoch              ).reshape(-1,1),
                               np.array(history.history['mae']     ).reshape(-1,1),
                               np.array(history.history['val_mae'] ).reshape(-1,1),
                               np.array(history.history['loss']    ).reshape(-1,1),
                               np.array(history.history['val_loss']).reshape(-1,1)])
    history_df = pd.DataFrame(history_table, columns=['epoch','mae','val_mae','loss','val_loss'])
    history_df.to_csv(history_path, index=False)


def yn(str):
    # TODO: move to data module
    if str.lower() in ['y', 'yes', 'yea', 'ok', 'okay', 'k',  
                       'sure', 'hai', 'aye', 'ayt', 'fosho']:
        return True
    elif str.lower() in ['n', 'no', 'nope', 'nah', 'hold this l']:
        return False
    else:
        raise Exception(str + 'not recognized: use y - yes, n - no')


def read_aug_data(file):
    # TODO: move to data module
    """Read data file and return a DataFrame.

    Args:
        file (PosixPath): Path to .feather file.

    Returns:
        interp_df: DataFrame of interpolated data.
    """
    interp_df = pd.read_feather(file).drop(columns=['Ex (V/m)', 'Ey (V/m)'])
    return interp_df


##### data processing ######
def get_augmentation_data(data_used, xy: bool, vp: bool):
    # TODO: move to data module
    """Get augmentation data for training.

    Args:
        xy (bool): Include xy grid augmented data
        vp (bool): Inclde vp augmentation data
    """
    if xy:  # xy augmentation
        xyfile = Path(root/'data'/'interpolation_datasets'/'rec-interpolation2.nc')
        xydf = xr.open_dataset(xyfile).to_dataframe().reset_index().dropna()
    else: xydf = None

    if vp:  # vp augmentation
        vpfolder = Path(root/'data'/'interpolation_feather'/'20221209')
        # read all files and combine into a single df
        vpdf = [read_aug_data(file) for file in vpfolder.glob('*.feather')]
        vpdf = pd.concat(vpdf).rename(columns={'Vpp [V]' : 'V', 
                                               'P [Pa]'  : 'P',
                                               'X'       : 'x',
                                               'Y'       : 'y'}, inplace=True)
    else: vpdf = None

    # make sure that the data follows the correct format before returning
    assert list(data_used.columns) == ['V', 'P', 'x', 'y'] + label_names
    return pd.concat([data_used, xydf, vpdf], ignore_index=True)
    

def get_data(xy=False, vp=False):
    # TODO: move to data module
    """Get dataset

    Assumes the DataFrame has previously been saved as a .feather file. If not,
    a new .feather file is created in the root/data folder.

    Args:
        xy (bool, optional): Include xy augmentation. Defaults to False.
        vp (bool, optional): Include vp augmentation. Defaults to False.

    Raises:
        Exception: Raises an error if no data is available in avg_data.

    Returns:
        data_used: DataFrame of data to be used, with specified augmentation data.
        data_excluded: DataFrame of excluded data.
    """
    avg_data_file = root/'data'/'avg_data.feather'

    # check if feather file exists and load avg_data
    if avg_data_file.is_file():
        print('reading data feather file...')
        avg_data = pd.read_feather(avg_data_file)
    
    else:
        print('data feather file not found, building file...')
        start_time = time.time()
        avg_data = data.read_all_data(data_fldr_path, voltages, pressures).drop(columns=['Ex (V/m)', 'Ey (V/m)'])
        elapsed_time = time.time() - start_time
        print(f' done ({elapsed_time:0.1f} sec).\n')

        if len(avg_data)==0:
            raise Exception('No data available.')

        avg_data.to_feather(avg_data_file)

    # separate data to be excluded (to later check the model)
    data_used     = avg_data[~((avg_data['Vpp [V]']==voltage_excluded) & (avg_data['P [Pa]']==pressure_excluded))].copy()
    data_excluded = avg_data[  (avg_data['Vpp [V]']==voltage_excluded) & (avg_data['P [Pa]']==pressure_excluded) ].copy()

    # rename columns
    data_used.rename(columns={'Vpp [V]' : 'V',
                              'P [Pa]'  : 'P',
                              'X'       : 'x', 
                              'Y'       : 'y'}, inplace=True)

    if (xy or vp):
        data_used = get_augmentation_data(data_used, xy, vp)

    # create new column of x^2 and y^2
    data_used['x**2'] = data_used['x']**2
    data_used['y**2'] = data_used['y']**2
    
    return data_used, data_excluded


# --------------- Model hyperparameters -----------------
# model name
name = input('Enter model name: ') or 'test'

root = Path(os.getcwd())
data_fldr_path = root/'data'/'avg_data'

# training inputs
batch_size = int(input('Batch size (default 128): ') or '128')
learning_rate = float(input('Learning rate (default 0.001): ') or '0.001')
validation_split = float(input('Validation split (default 0.1): ') or '0.1')
epochs = int(input('Epochs (default 5): ') or '5')
minmax_y = True # not args['unscaleY']  # opposite of args[unscaleY], i.e.: False if unscaleY flag is raised
lin = True # not args['log']  # opposite of args[log], i.e.: False if log flag is raised

# -------------------------------------------------------

voltages  = [200, 300, 400, 500] # V
pressures = [  5,  10,  30,  45, 60, 80, 100, 120] # Pa


voltage_excluded = 300 # V
pressure_excluded = 60 # Pa

# -------------------------------------------------------

# TODO: include test modifications
out_dir = create_output_dir()
scaler_dir = out_dir / 'scalers'
os.mkdir(scaler_dir)

# copy some files for backup (probably made redundant by metadata)
shutil.copyfile(__file__, out_dir / 'create_model.py')
shutil.copyfile(root / 'data.py', out_dir / 'data.py')

feature_names = ['V', 'P', 'x', 'y']
label_names = ['potential (V)', 'Ne (#/m^-3)', 'Ar+ (#/m^-3)', 'Nm (#/m^-3)', 'Te (eV)']

data_used, data_excluded = get_data(xy=False, vp=True)

# set threshold to make very small values zero
pd.set_option('display.chop_threshold', 1e-10)

# scale features and labels
scale_exp = []
features = scale_all(data_used[feature_names], 'x', scaler_dir).astype('float64')
labels = data_preproc(data_used[label_names]).astype('float64')

if minmax_y:  # if applying minmax to target data
    labels = scale_all(labels, 'y', scaler_dir)

alldf = pd.concat([features, labels], axis=1)
dataset_size = len(alldf)

# create dataset object and shuffle it()  # TODO: train/val split
trainset = torch.utils.data.TensorDataset(features.to_numpy(), labels.to_numpy())
trainloader = torch.utils.data.DataLoader(trainset, 
                                          batch_size=batch_size, 
                                          shuffle=True,
                                          num_workers=2)

# determine validation split


# create validation split and batch the data

model = MLP(name, len(feature_names), len(label_names)) 
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train the model
print('begin model training...')
train_start = time.time()  # record start time

for epoch in tqdm(range(epochs)):

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
        # if i % 4 == 3:  # print every 4 mini batches (?)
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss/4:.3f}\r')
        running_loss = 0.0

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
# TODO: record epoch times in times
with open(out_dir / 'times.txt', 'w') as f:
    f.write('Train times per epoch\n')
    for i, time in enumerate(times):
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
    f.write(f'Average time per epoch: {np.array(times).mean():.2f} s\n')
    f.write(f'\nUser-specified hyperparameters\n')
    f.write(f'Batch size: {batch_size}\n')
    f.write(f'Learning rate: {learning_rate}\n')
    f.write(f'Validation split: {validation_split}\n')
    f.write(f'Epochs: {epochs}\n')
    f.write('\n*** end of file ***\n')
