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


def scale_targets(data_table: pd.DataFrame, label_names: list, scale_exp=[1.0, 14.0, 14.0, 16.0, 0.0]):
    """Scale target data (from simulation results).

    Target data is in original scaling, whereas the training predictions are
    scaled down by 10^n. Exponents used to scale the data down for 
    training (n) are stored as a pickle file in the model folder. 
    The pickle file is a list of powers, which are used to reverse the scaling.

    Args:
        data_table (pd.DataFrame): DataFrame of target data.
        label_names (list): List of column names to scale.
        scale_exp (list, optional): List of parameter exponents. 
            Defaults to [1.0, 14.0, 14.0, 16.0, 0.0].

    Returns:
        pd.DataFrame: DataFrame of scaled target data.
    """

    # label_names = ['potential (V)', 'Ne (#/m^-3)', 'Ar+ (#/m^-3)', 'Nm (#/m^-3)', 'Te (eV)']
    for i, column in enumerate(label_names):
        data_table.update(data_table.iloc[:, i]/(10**scale_exp[i]))

    return data_table


def reverse_minmax(df: pd.DataFrame, columns: list, model_dir: Path):
    """Reverse minmax scaling on data.

    Model predictions are minmax-scaled to lie between 0 and 1. This function
    reverses it for comparison with simulation data.

    Args:
        df (pd.DataFrame): _description_
        columns (list): _description_
        model_dir (Path): _description_

    Returns:
        _type_: _description_
    """

    def reverse_minmax_core(values, n):
        values = np.array([values]).reshape(1, -1)
        scaler_file = model_dir/'scalers'/f'yscaler_{n:02}.pkl'
        with open(scaler_file, 'rb') as sf:
            yscaler = pickle.load(sf)
            yscaler.clip = False
            return yscaler.inverse_transform(values).item()

    columns = [df[column].apply(reverse_minmax_core, args=(n,)) for n, column in enumerate(df.columns)]
    scaled_df = pd.concat(columns, axis=1)

    return scaled_df


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
    regr_df['V'] = 300
    regr_df['P'] = 60
    regr_df.rename(columns={'X':'x', 'Y':'y'}, inplace=True)
    regr_df = regr_df[feature_names + label_names]  # fix arrangement

    features = regr_df.drop(columns=label_names)
    labels = regr_df[label_names]

    # infer regression details from model metadata
    if os.path.exists(model_dir / 'train_metadata.pkl'):
        with open(model_dir / 'train_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        minmax_y = metadata['is_target_scaled']
        lin = metadata['scaling']
        name = metadata['name']
        scale_exp = metadata['parameter_exponents']
    else:
        print('Metadata unavailable: using defaults lin=True, minmax_y=True\n')
        lin = True
        minmax_y = True
        name = model_dir.name
    
        print('\nLoaded model ' + name)

    features_tensor = scale_features(features, model_dir)
    targets = scale_targets(labels, label_names, scale_exp)

    model = MLP(len(feature_names), len(label_names))
    model.load_state_dict(torch.load(model_dir/f'{name}'))
    
    model.eval()  # set model to eval mode
    model(features_tensor)
    scaled_prediction = pd.DataFrame(model(features_tensor).detach().numpy(), columns=label_names)  # get the prediction
    
    # pred_df = pd.concat([features[['x', 'y']], scaled_prediction])

    # plot the predictions
    import matplotlib
    def triangulate(df):   
        """
        Create triangulation of the mesh grid, which is passed to tricontourf.
        
        Uses Delaunay triangulation.

        Parameters
        ----------
        df : DataFrame
            DataFrame with X and Y values for the triangulation.

        Returns
        -------
        triangles : matplotlib.tri.triangulation.Triangulation
            Triangulated grid.

        """
        x = df['x'].to_numpy()*100
        y = df['y'].to_numpy()*100
        triangles = matplotlib.tri.Triangulation(x, y)
        
        return triangles

    triangles = triangulate(features[['x', 'y']])
    n = 1
    fig = plt.figure(figsize=(6, 8))
    fig.suptitle('Autoencoder predicting potential (V)')
    # for i in range(n):
    #     # display original
    #     ax = plt.subplot(n, 2, i + 1)
    #     im = plt.imshow(v_test[0,:,:,0], origin='lower', vmin=0, vmax = 0.4)
    #     plt.title("original")
    #     plt.gray()
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)

    #     # display reconstruction
    #     ax = plt.subplot(n, 2, i + 1 + n)
    #     plt.imshow(decoded_imgs[0,:,:,0], origin='lower', vmin=0, vmax = 0.4)
    #     plt.title("reconstructed")
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)
    fig.colorbar(im)
    plt.show()
    