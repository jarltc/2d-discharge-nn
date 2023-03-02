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
import plot


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


class PredictionDataset:
    def __init__(self, reference_df, model, metadata) -> None:
        self.original_df = reference_df
        self.model = model
        self.v_excluded = 300  # [V]
        self.p_excluded = 60  # [Pa]
        self.df, self.features, self.labels = \
            process_data(reference_df, (self.v_excluded, self.p_excluded))
        self.targets = scale_targets(self.labels, self.scale_exp)
        self.minmax_y = metadata['is_target_scaled']
        self.lin = metadata['scaling']
        self.scale_exp = metadata['parameter_exponents']

    def make_prediction(self):
         features_tensor = scale_features(self.df, model_dir)

         model.eval()
         prediction = pd.DataFrame(model(features_tensor)\
                                          .detach().numpy(), 
                                          columns=list(self.labels.columns))
         
         prediction = reverse_minmax(prediction, model_dir)
         return prediction
    
    def print_scores(self, prediction):
        r2, mae, rmse, ratio = get_scores(prediction)
        data = np.vstack([r2, mae, rmse, ratio])
        scores_df = pd.DataFrame(data, columns=list(self.labels.columns))
        
        # TODO: print to stdout and save to txt file
        for column in scores_df.columns:
            print(f'**** {column} ****')
            print(f'MAE = {scores_df[column,1]}')
            print(f'RMSE = {scores_df[column,2]}')
            print(f'RMSE/MAE = {scores_df[column,3]}\n')
            print(f'R2 = {scores_df[column,0]}')

        with open('') as file:
            file.write
        

def process_data(df: pd.DataFrame, data_excluded: tuple):
    v_excluded, p_excluded = data_excluded
    df['V'] = v_excluded
    df['P'] = p_excluded
    df = df[feature_names + label_names]
    df.rename(columns={'X':'x', 'Y':'y'}, inplace=True)

    features = df.drop(columns=label_names)
    labels = df[label_names]

    return df, features, labels


def scale_features(df: pd.DataFrame, model_dir: Path):
    """Scale table's features for regression.

    Args:
        df (pd.DataFrame): Table of features to scale.
        model_dir (Path): Model directory to get the scaler files.

    Returns:
        torch.FloatTensor: Tensor of features that can be directly used for
        model evaluation.
    """
    def scale_features_core(values, n):
        # column_values = df[column].values.reshape(-1, 1)
        values = np.array([values]).reshape(1, -1)
        scaler_file = model_dir/'scalers'/f'xscaler_{n:02}.pkl'
        with open(scaler_file, 'rb') as sf:
            xscaler = pickle.load(sf)
            xscaler.clip = False
            return xscaler.transform(values).item()

    columns = [df[column].apply(scale_features_core, args=(n,)) for n, column in enumerate(df.columns, start=1)]
    scaled_df = pd.concat(columns, axis=1)

    # columns = [scale_features_core(n, column) for n, column in enumerate(data_table.columns, start=1)]
    # scaled_df = pd.DataFrame(np.hstack(columns), columns=data_table.columns)

    return torch.FloatTensor(scaled_df.to_numpy())


def scale_targets(data_table: pd.DataFrame, scale_exp=[1.0, 14.0, 14.0, 16.0, 0.0]):
    """Scale target data (from simulation results).

    Target data is in original scaling, and will be scaled down to match
    the prediction data. Exponents used to scale the data down for 
    training (n) are stored as a pickle file in the model folder. 
    The pickle file is a list of powers which are used to reverse the scaling.

    Args:
        data_table (pd.DataFrame): DataFrame of target data.
        label_names (list): List of column names to scale.
        scale_exp (list, optional): List of parameter exponents. 
            Defaults to [1.0, 14.0, 14.0, 16.0, 0.0].

    Returns:
        pd.DataFrame: DataFrame of scaled target data.
    """
    label_names = list(data_table.columns)
    # label_names = ['potential (V)', 'Ne (#/m^-3)', 'Ar+ (#/m^-3)', 'Nm (#/m^-3)', 'Te (eV)']
    for i, _ in enumerate(label_names):
        unscaled_df = data_table.iloc[:, i]/(10**scale_exp[i])

    return unscaled_df


def reverse_minmax(df: pd.DataFrame, model_dir: Path):
    """Reverse minmax scaling on data.

    Model predictions are minmax-scaled to lie between 0 and 1. This function
    reverses it for comparison with simulation data.

    Args:
        df (pd.DataFrame): DataFrame of scaled predictions.
        model_dir (Path): Path to model directory where scalers are stored.

    Returns:
        pd.DataFrame: DataFrame of unscaled predictions.
    """

    def reverse_minmax_core(values, n):
        values = np.array([values]).reshape(1, -1)
        scaler_file = model_dir/'scalers'/f'yscaler_{n:02}.pkl'
        with open(scaler_file, 'rb') as sf:
            yscaler = pickle.load(sf)
            yscaler.clip = False
            return yscaler.inverse_transform(values).item()

    columns = [df[column].apply(reverse_minmax_core, args=(n,)) for n, column in enumerate(df.columns, start=1)]
    scaled_df = pd.concat(columns, axis=1)

    return scaled_df


def get_scores(reference_df: pd.DataFrame, prediction_df: pd.DataFrame):
    y_true = reference_df.values
    y_pred = prediction_df.values

    # compute regression scores
    r2 = (np.abs(y_true - y_pred)**2).mean(axis=0)  
    mae = np.abs(y_true - y_pred).mean(axis=0)  
    rmse = np.sqrt(r2)

    return r2, mae, rmse, rmse/mae

if __name__ == '__main__':
    feature_names = ['V', 'P', 'x', 'y']
    label_names = ['potential (V)', 'Ne (#/m^-3)', 'Ar+ (#/m^-3)', 'Nm (#/m^-3)', 'Te (eV)']

    root = Path.cwd()
    model_dir = Path(input('Model directory: ') or './created_models/test_dir_torch')
    regr_df = data.read_file(root/'data'/'avg_data'/'300Vpp_060Pa_node.dat')\
        .drop(columns=['Ex (V/m)', 'Ey (V/m)'])
    regr_df = PredictionDataset(regr_df)

    # infer regression details from model metadata, else assume defaults
    if os.path.exists(model_dir / 'train_metadata.pkl'):
        with open(model_dir / 'train_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
            name = metadata['name']
    else:
        print('Metadata unavailable: using defaults lin=True, minmax_y=True\n')
        metadata = {'scaling': True, 'is_target_scaled':True, 'name':model_dir.name}

    model = MLP(len(feature_names), len(label_names))
    model.load_state_dict(torch.load(model_dir/f'{name}'))
    print('\nLoaded model ' + name)

    prediction = regr_df.make_prediction()

    triangles = plot.triangulate(regr_df.features[['x', 'y']])
    plot.quickplot(prediction, model_dir, triangles=triangles)
    