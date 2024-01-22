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
from do_regr import process_data, scale_features
from time import sleep

from sklearn.metrics import mean_squared_error, r2_score


def scores(original:pd.DataFrame, prediction:pd.DataFrame):
    # get the MSE between each column of original and predicted dataframes
    global label_names

    orig_labels = original[label_names]
    pred_labels = prediction[label_names]

    mse = []
    r2 = []

    for column in label_names:
        y_true = orig_labels[column].values
        y_pred = pred_labels[column].values

        mse.append(mean_squared_error(y_true, y_pred))
        r2.append(r2_score(y_true, y_pred))
    
    return mse, r2


def load_data(test_pair:tuple):
    global root, hps, feature_names, label_names, scale_exp
    batch_size = hps['batch_size']

    # this data is not yet scaled
    data_used, data_excluded = get_data(root, voltages, pressures, test_pair, xy=True)
    
    # why the hell do I have to do so much
    # scale features and labels
    features = scale_all(data_used[feature_names], 'x').astype('float64')
    labels = data_preproc(data_used[label_names], scale_exp).astype('float64')
    labels = scale_all(labels, 'y')

    features = torch.tensor(features.to_numpy(), dtype=torch.float64)
    labels = torch.tensor(labels.to_numpy(), dtype=torch.float64)
    dataset = TensorDataset(features, labels)
    trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return trainloader, data_excluded


def train_mlp(train_data:DataLoader):
    global hps
    name = hps['name']
    epochs = hps['epochs']
    learning_rate = hps['learning_rate']
    model = MLP(name, 4, 5)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print('Training model...')
    sleep(5)
    # # actually train the model
    # model.train()  # is this even needed?
    # for epoch in tqdm(range(epochs), desc='Training...', colour='#7dc4e4'):
    #     loop = tqdm(train_data, unit='batch', colour='#f5a97f')
    #     for i, batch_data in enumerate(loop):
    #         inputs, labels = batch_data
    #         loop.set_description(f'batch {i} of {len(loop)}')

    #         optimizer.zero_grad()  # zero the parameter gradients
            
    #         outputs = model(inputs)  # forward pass
    #         loss = criterion(outputs, labels)  # compute score
    #         loss.backward()  # compute gradients 
    #         optimizer.step()  # apply changes to network

    return model


def model_eval(model, data_excluded, test_pair):
    global feature_names, label_names, scale_exp
    model.eval()

    # load dummy data (only grid points and labels, no v, p)
    reference_df = data_excluded
    regr_df, features_df, _ = process_data(reference_df, test_pair)

    # scale features then convert into a tensor TODO: add model dir to get scaler files
    features_tensor = scale_features(features_df, model_dir)
    
    # feed features to model and get tensor of outputs
    with torch.no_grad:  # i wonder if this speeds it up somewhat
        result = pd.DataFrame(model(features_tensor).detach().numpy(),
                            columns=label_names)
    
    # combine old features into results dataframe
    prediction_df = pd.concat([result, features_df], axis=1)

    # compare scaled labels and outputs by getting mse and r2
    mse, r2 = scores(regr_df, prediction_df)
    return mse, r2
    

if __name__ == "__main__":
    voltages = [300.0, 400.0]
    pressures = [30.0, 45.0, 60.0, 80.0, 100.0]
    vps = [(v, p) for v in voltages for p in pressures]
    feature_names = ['V', 'P', 'x', 'y']
    label_names = ['potential (V)', 'Ne (#/m^-3)', 'Ar+ (#/m^-3)', 'Nm (#/m^-3)', 'Te (eV)']

    # hyperparameters TODO: change
    scale_exp = []
    hps = {'name':'kfold', 'epochs':1, 'batch_size':256, 'learning_rate': 1e-3}

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
            model = train_mlp(data)
            r2, mse = model_eval(model, data_excluded, vp)
