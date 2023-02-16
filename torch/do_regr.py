"""
Model regression code. (PyTorch)

created: @jarl on 16 Feb 14:52
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd

import os
import pickle
from pathlib import Path

import data


class MLP(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super(MLP, self).__init__()
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


def scale_features(data_table: pd.DataFrame, model_dir: Path):
    """Scale table's features for regression.

    Args:
        data_table (pd.DataFrame): Table of features to scale.
        model_dir (Path): Model directory to get the scaler files.

    Returns:
        torch.FloatTensor: Tensor of features that can be directly used for
        model evaluation.
    """
    def scale_features_core(n, column):
        column_values = data_table[column].values.reshape(-1, 1)
        scaler_file = model_dir/'scalers'/f'xscaler_{n:02}.pkl'
        with open(scaler_file, 'rb') as sf:
            xscaler = pickle.load(sf)
            xscaler.clip = False
            return xscaler.transform(column_values)

    columns = [scale_features_core(n, column) for n, column in enumerate(data_table.columns, start=1)]
    scaled_df = pd.DataFrame(np.hstack(columns))

    return torch.FloatTensor(scaled_df.to_numpy())


def scale_labels(data_table: pd.DataFrame, model_dir: Path, lin=True):
    """Scale table's labels for regression.

    Args:
        data_table (pd.DataFrame): Table of labels to scale.
        model_dir (Path): Model directory to get the scaler files.
        lin (bool, optional): Scale inputs linearly. Defaults to True.
    """
    def scale_features_core(n, column):
        column_values = data_table[column].values.reshape(-1, 1)
        scaler_file = model_dir/'scalers'/f'xscaler_{n:02}.pkl'
        with open(scaler_file, 'rb') as sf:
            xscaler = pickle.load(sf)
            xscaler.clip = False
            return xscaler.transform(column_values)

    columns = [scale_features_core(n, column) for n, column in enumerate(data_table.columns, start=1)]
    scaled_df = pd.DataFrame(np.hstack(columns))

    return torch.FloatTensor(scaled_df.to_numpy())


def get_scores():
    # compute regression scores
    return r2, mae, rmse, rmse/mae

if __name__ == '__main__':
    feature_names = ['V', 'P', 'x', 'y']
    label_names = ['potential (V)', 'Ne (#/m^-3)', 'Ar+ (#/m^-3)', 'Nm (#/m^-3)', 'Te (eV)']

    root = Path.cwd()
    model_dir = Path(input('Model directory: '))
    regr_df = data.read_file(root/'data'/'avg_data'/'300Vpp_060Pa_node.dat')\
        .drop(columns=['Ex (V/m)', 'Ey (V/m)'])
    features = regr_df.drop(columns=label_names)
    labels = regr_df[label_names]

    # infer regression details from model metadata
    if os.path.exists(model_dir / 'train_metadata.pkl'):
        with open(model_dir / 'train_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        minmax_y = metadata['is_target_scaled']
        lin = metadata['scaling']
        name = metadata['name']
    else:
        print('Metadata unavailable: using defaults lin=True, minmax_y=True\n')
        lin = True
        minmax_y = True
        name = model_dir.name
    
        print('\nLoaded model ' + name)

    features_tensor = scale_features(features, model_dir)
    labels_tensor = scale_labels(labels, model_dir, lin)
    model = MLP(len(feature_names), len(label_names))
    model.load_state_dict(torch.load(model_dir/f'{name}'))
    model.eval()
    # model
