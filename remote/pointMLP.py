import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.benchmark as benchmark
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pandas as pd

import time
import shutil
import yaml
from pathlib import Path
from tqdm import tqdm

import pointmlp_classes as pointMLP
from data_helpers import set_device, get_data, scale_all, data_preproc

def get_input_file(root):
    input_dir = root/'inputs'/'pointmlp'
    input_files = [in_file.stem for in_file in input_dir.glob("*.yml")]
    in_name = input(f"Choose input file for training (leave blank for default.yml)\n{input_files}\n> ")
    
    if in_name:  # if not empty
        in_file = input_dir/f"{in_name}.yml"
        if not in_file.exists():
            raise ValueError(f"{in_name} is not a recognized input file.")
        else:
            return in_file
    else:
        return input_dir/"default.yml"


def read_input(input_file: Path):

    def from_yaml(filename):
        """
        Load train parameters from a YAML file.

        Args:
        filename (Path): Path to the YAML file to load the data from.

        Returns:
        dict: The loaded metadata.
        """
        with open(filename, 'r') as yaml_file:
            data = yaml.safe_load(yaml_file)
        
        return data
        
    data = from_yaml(input_file)
    hyperparameters = data['hyperparameters']
    del data['hyperparameters']

    parameters = data

    return parameters, hyperparameters


def load_model(parameters:dict, input_size=4, output_size=5):
    device = set_device()
    mlp = pointMLP.get_model(parameters['model'], input_size, output_size).to(device)
    return mlp 


def set_hyperparameters(model: nn.Module, hyperparameters: dict):

    learning_rate = hyperparameters['learning_rate']
    
    if hyperparameters['criterion'] == "MSE":
        hyperparameters['criterion'] = nn.MSELoss()
    else:
        raise ValueError(f"{hyperparameters['criterion']} is not recognized")

    if hyperparameters['optimizer'] == "Adam":
        hyperparameters['optimizer'] = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"{hyperparameters['optimizer']} is not recognized")
    
    return hyperparameters


if __name__ == "__main__":
    device = set_device()

    # get parameters from input file
    root = Path.cwd()
    input_file = get_input_file(root)
    params, hyperparameters = read_input(input_file)

    # load model
    mlp = load_model(params)

    out_dir = mlp.path
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
    shutil.copy(str(input_file), str(out_dir/'input.yml'))  # make a copy of the input file

    # initialize hyperparameters
    hyperparameters = set_hyperparameters(mlp, hyperparameters)
    name = mlp.name
    test_pair = (params['testV'], params['testP'])

    feature_names = ['V', 'P', 'x', 'y']
    label_names = ['potential (V)', 'Ne (#/m^-3)', 'Ar+ (#/m^-3)', 'Nm (#/m^-3)', 'Te (eV)']
    
    # TODO: infer these from a dataset
    voltages  = [200, 300, 400, 500] # V
    pressures = [  5,  10,  30,  45, 60, 80, 100, 120] # Pa

    data_used, data_excluded = get_data(root, voltages, pressures,
                                        test_pair, xy=hyperparameters['grid_aug'], 
                                        vp=hyperparameters['vp_aug'])

    #### data preprocessing
    pd.set_option('display.chop_threshold', 1e-10)  # TODO: check if needed
    
    scale_exp = []
    scaler_dir = out_dir / 'scalers'
    if (not scaler_dir.exists()):
        scaler_dir.mkdir(parents=True)

    # TODO: remake these
    features = scale_all(data_used[feature_names], 'x', scaler_dir).astype('float64')
    labels = data_preproc(data_used[label_names], scale_exp).astype('float64')
    if params['minmax']:
        labels = scale_all(labels, 'y', scaler_dir)

    # TODO: add validation split
    features = torch.tensor(features.to_numpy(), device=device, dtype=torch.float32)  # TODO: make dtype a global parameter
    labels = torch.tensor(labels.to_numpy(), device=device, dtype=torch.float32)
    dataset = TensorDataset(features, labels)
    trainloader = DataLoader(dataset, batch_size=hyperparameters['batch_size'],
                             shuffle=True)

    criterion = hyperparameters['criterion']
    optimizer = hyperparameters['optimizer']
    epochs = hyperparameters['epochs']
    
    # TODO: track metrics (epoch, validation losses)
    mlp.train()
    start = time.perf_counter()
    
    #### main training loop
    for epoch in tqdm(range(epochs)):
        loop = tqdm(trainloader)
        for i, batch_data in enumerate(loop):
            inputs, labels = batch_data

            optimizer.zero_grad()
            running_loss = 0.0

            outputs = mlp(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # compute gradients
            optimizer.step()  # apply changes to network
    
    end = time.perf_counter()
    torch.save(mlp.state_dict, out_dir/'model.pt')
    print(f'Completed {epochs} epochs in {end-start:.2f} s')
