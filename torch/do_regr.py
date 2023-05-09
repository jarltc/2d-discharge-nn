"""
Model regression code. (PyTorch)

created: @jarl on 16 Feb 14:52
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import os
import sys
import pickle
from pathlib import Path

import data_helpers
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


class MLP2(nn.Module):
    """Neural network model for grid-wise prediction of 2D-profiles.

    Model architecture optimized using OpTuna + 2 more layers.
    Args:
        name (string): Model name
        input_size (int): Size of input vector.
        output_size (int): Size of output vector. Should be 5 for the usual variables.
    """
    def __init__(self, input_size, output_size) -> None:
        super(MLP2, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, 115)  # linear: y = Ax + b
        self.fc2 = nn.Linear(115, 78)
        self.fc3 = nn.Linear(78, 46)
        self.fc4 = nn.Linear(46, 26)
        self.fc5 = nn.Linear(26, 46)
        self.fc6 = nn.Linear(46, 82)
        self.fc7 = nn.Linear(82, 106)
        self.fc8 = nn.Linear(106, 115)
        self.fc9 = nn.Linear(115, output_size)
        
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
        x = F.relu(x)
        x = self.fc8(x)
        x = F.relu(x)
        x = self.fc9(x)
        
        output = x = F.relu(x)
        return output


class PredictionDataset:
    """
    A dataset for making predictions.

    Attributes
    ----------
    original_df : pd.DataFrame
        The original reference DataFrame.
    model : torch.nn.Module
    model_name : str
    v_excluded : float
        Voltage excluded during training.
    p_excluded :
        Pressure excluded during training.
    minmax_y : bool
        Whether targets were scaled for training.
    lin : bool
        Whether targets were linearly scaled before minmax.
    scale_exp : list of float
        Parameter exponents for linear scaling.
    df : pd.DataFrame
        DataFrame after adding new V, P columns renaming and reordering.
    features : pd.DataFrame
        DataFrame of dataset features.
    labels : pd.DataFrame
        DataFrame of dataset labels.
    targets : pd.DataFrame
        DataFrame of labels after scaling. Used for computing scores.
    prediction_result : pd.DataFrame
        DataFrame of model prediction on self.features.
    scores : pd.DataFrame
        DataFrame of scores computed in calculate_scores().
    
    Methods
    -------
    prediction():
        Makes a prediction on a model's features.
    get_scores():
        Computes scores, outputs to sys.stdout and saves to a txt file.
    """
    def __init__(self, reference_df, model, metadata) -> None:
        self.original_df = reference_df
        self.model = model
        self.model_name = metadata['name']
        self.v_excluded = 300  # [V]
        self.p_excluded = 60  # [Pa]
        self.minmax_y = metadata['is_target_scaled']
        self.lin = metadata['scaling']
        self.scale_exp = metadata['parameter_exponents']
        self.df, self.features, self.labels = \
            process_data(reference_df, (self.v_excluded, self.p_excluded))
        self.targets = scale_targets(self.labels, self.scale_exp)
        self.prediction_result = None
        self.scores = None

    @property
    def prediction(self):
        self.model.eval()
        if self.prediction_result is None:
            print(f"\nGetting {self.model_name} prediction...\r", end="")
            features_tensor = scale_features(self.features, model_dir)

            result = pd.DataFrame(self.model(features_tensor)\
                                            .detach().numpy(), 
                                            columns=list(self.labels.columns))
            
            result = reverse_minmax(result, model_dir)
            print("\33[2KPrediction complete!")
            self.prediction_result = result
            return result
        else:
            print("Prediction result already calculated.")
            return self.prediction_result

    def get_scores(self):
        scores_df = calculate_scores(self.targets, self.prediction_result)
        self.scores = scores_df
        
        def print_scores_core(out):
            for column in scores_df.columns:
                print(f'**** {column} ****', file=out)
                print(f'MAE\t\t= {scores_df[column].iloc[0]}', file=out)
                print(f'RMSE\t\t= {scores_df[column].iloc[1]}', file=out)
                print(f'RMSE/MAE\t= {scores_df[column].iloc[2]}', file=out)
                print(f'R2\t\t= {scores_df[column].iloc[3]}', file=out)
                print(file=out)

        print_scores_core(sys.stdout)
        plot.correlation(self.targets, self.prediction_result, self.scores, regr_dir)

        scores_file = regr_dir/'scores.txt'
        with open(scores_file, 'w') as f:
            print_scores_core(f)

        

def process_data(df: pd.DataFrame, data_excluded: tuple):
    v_excluded, p_excluded = data_excluded
    df['V'] = v_excluded
    df['P'] = p_excluded
    df.rename(columns={'X':'x', 'Y':'y'}, inplace=True)
    df = df[feature_names + label_names]

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
    scaled_df = data_table.copy()
    for i, column in enumerate(data_table.columns):
        scaled_df[column].update(data_table.iloc[:, i]/(10**scale_exp[i]))

    return scaled_df


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


def calculate_scores(reference_df: pd.DataFrame, prediction_df: pd.DataFrame):
    """Calculate prediction scores.

    Args:
        reference_df (pd.DataFrame): Reference DataFrame, scaled to match prediction.
        prediction_df (pd.DataFrame): Prediction DataFrame.

    Returns:
        pd.DataFrame: DataFrame of scores. Rows are MAE, RMSE, RMSE/MAE, and R2.
    """
    scores = []
    for column in reference_df.columns:
        y_true = reference_df[column].values
        y_pred = prediction_df[column].values

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        ratio = rmse/mae
        scores.append(np.array([[mae], [rmse], [ratio], [r2]]))
    
    scores = np.hstack(scores)
    scores_df = pd.DataFrame(scores, columns=list(reference_df.columns))

    return scores_df

if __name__ == '__main__':
    feature_names = ['V', 'P', 'x', 'y']
    label_names = ['potential (V)', 'Ne (#/m^-3)', 'Ar+ (#/m^-3)', 'Nm (#/m^-3)', 'Te (eV)']

    # define directories
    root = Path.cwd()
    model_dir = Path(input('Model directory: ') or './created_models/test_dir_torch')
    regr_dir = model_dir / 'prediction'
    if not regr_dir.exists():
        os.mkdir(regr_dir)

    # infer regression details from training metadata, else assume defaults
    if os.path.exists(model_dir / 'train_metadata.pkl'):
        with open(model_dir / 'train_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
            name = metadata['name']
    else:
        print('Metadata unavailable: using defaults lin=True, minmax_y=True\n')
        metadata = {'scaling': True, 'is_target_scaled':True, 'name':model_dir.name}

    # import model
    model = MLP2(4, 5)
    model.load_state_dict(torch.load(model_dir/f'{name}'))
    print('\nLoaded model ' + name)

    # load dataset
    regr_df = data_helpers.read_file(root/'data'/'avg_data'/'300Vpp_060Pa_node.dat')\
        .drop(columns=['Ex (V/m)', 'Ey (V/m)'])
    regr_df = PredictionDataset(regr_df, model, metadata)

    prediction = regr_df.prediction  # make a prediction
    regr_df.get_scores()  # get scores

    triangles = plot.triangulate(regr_df.features[['x', 'y']])
    plot.quickplot(prediction, regr_dir, triangles=triangles)
    plot.difference_plot(regr_df.features, prediction, regr_df.targets, out_dir=regr_dir)
    