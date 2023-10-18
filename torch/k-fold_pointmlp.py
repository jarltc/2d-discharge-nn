""" k-fold validation of the pointMLP
"""

from pathlib import Path
from tqdm import tqdm
import datetime as dt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

from MLP import MLP
from data_helpers import get_data, scale_all, data_preproc
from do_regr import PredictionDataset


def load_data(test_pair:tuple):
    global root, hps, feature_names, label_names
    batch_size = hps['batch_size']
    
    data_used, data_excluded = get_data(root, voltages, pressures, test_pair, xy=True)
    
    # why the hell do I have to do so much
    scaler_dir = 100  # TODO: fix this
    # scale features and labels
    scale_exp = []
    features = scale_all(data_used[feature_names], 'x', scaler_dir).astype('float64')
    labels = data_preproc(data_used[label_names], scale_exp).astype('float64')
    labels = scale_all(labels, 'y', scaler_dir)

    alldf = pd.concat([features, labels], axis=1)  # TODO: consider removing this
    dataset_size = len(alldf)

    features = torch.tensor(features.to_numpy())
    labels = torch.tensor(labels.to_numpy())
    dataset = TensorDataset(features, labels)
    trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return trainloader, data_excluded


def train_mlp(train_data):
    global hps
    name = hps['name']
    epochs = hps['epochs']
    learning_rate = hps['learning_rate']
    model = MLP(name, 4, 5)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # actually train the model
    model.train()  # is this even needed?
    for epoch in tqdm(range(epochs), desc='Training...', colour='#7dc4e4'):
        for batch_data in enumerate(train_data):
            inputs, labels = batch_data

            optimizer.zero_grad()  # zero the parameter gradients
            
            outputs = model(inputs)  # forward pass
            loss = criterion(outputs, labels)  # compute score
            loss.backward()  # compute gradients 
            optimizer.step()  # apply changes to network

    return model


def model_eval(model, data_excluded):
    global feature_names, label_names
    model.eval()
    metadata = {'scaling': True, 'is_target_scaled':True, 'name': 'kfold'}
    regr_df = PredictionDataset(data_excluded, model, metadata)
    prediction = regr_df.prediction
    scores = regr_df.get_scores()  # dataframe of scores: rows are mae, rmse, rmse/mae, and r2
    mse = [scores.iloc[4, column] for column in range(5)]
    r2 = [scores.loc[3, column] for column in range(5)]

    return mse, r2
    

if __name__ == "__main__":
    voltages = [300.0, 400.0]
    pressures = [30.0, 45.0, 60.0, 80.0, 100.0]
    vps = [(v, p) for v in voltages for p in pressures]
    feature_names = ['V', 'P', 'x', 'y']
    label_names = ['potential (V)', 'Ne (#/m^-3)', 'Ar+ (#/m^-3)', 'Nm (#/m^-3)', 'Te (eV)']


    # hyperparameters
    hps = {'name':'kfold', 'epochs':100, 'batch_size':128, 'learning_rate': 1e-3}

    root = Path.cwd()
    out_dir = root/'kfold'
    if not out_dir.exists():
        out_dir.mkdir()

    loop = tqdm(vps, desc='k-fold validation')
    now = dt.datetime.now().strftime('%Y-%m-%d_%H%M')
    filename = now + '.txt'

    with open(out_dir/filename, 'a') as file:
        file.write(f'k-fold validation for k = {len(vps)}\n')
        file.write(f'voltages: {str(vps)}\n')
        file.write(f'pressures: {str(pressures)}\n')
        for vp in loop:
            description = f'test case {vp}'
            loop.set_description(description)
            file.write(f'results for excluded set {vp}:\n')

            data, data_excluded = load_data(vp)
            model = train_mlp(vp)
            r2, mse = model_eval(model, data_excluded)

