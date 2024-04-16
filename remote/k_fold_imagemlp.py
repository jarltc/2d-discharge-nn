"""
K-fold validation for images

To do:
* train a model based on excluded test set
* record validation losses after training each model

"""

import shutil
import time
from datetime import datetime as dt
from pathlib import Path
import yaml
from itertools import product

import matplotlib.pyplot as plt

import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision.transforms.functional import crop
from torch.utils.data import TensorDataset, DataLoader
import torch.utils.benchmark as benchmark
from torchinfo import summary

from data_helpers import ImageDataset, train2db, set_device
from plot import image_compare, save_history_graph, ae_correlation
import autoencoder_classes as AE
import mlp_classes
from image_data_helpers import get_data


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
    # TODO: change this to use pointMLP style
    hyperparameters = data['hyperparameters']
    parameters = {'model': data['model'],
                  'autoencoder': data['autoencoder'],
                  'random_seed': data['random_seed']}

    return parameters, hyperparameters


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


def load_models(parameters: dict):
    """ load models as specified in input file """
    device = set_device()
    autoencoder = AE.get_model(parameters['autoencoder']).eval()  # load autoencoder in inference mode
    autoencoder.load_state_dict(torch.load(autoencoder.path/'model.pt', map_location=device))  # load parameters

    input_size = 2  # model takes two input values
    x, y, z = autoencoder.encoded_size
    output_size = x*y*z
    
    model = mlp_classes.get_model(parameters['model'], input_size, output_size)
    return model, autoencoder


def get_input_file(root):
    input_dir = root/'inputs'/'k-fold'
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


def write_metadata(mlp, out_dir):
    name = mlp.name

    # save model structure
    file = out_dir/'kfold_log.txt'
    with open(file, 'w') as f:
        # model description
        f.write(f'k-fold validation of model {mlp.name}')
        f.write('(inputs are saved in the same folder)')

        # training details
        kfold = list(product(voltages, pressures))
        f.write(f'Sets: {kfold}')
        f.write('\n***** end of file *****')

def load_data(mlp:torch.nn.Module, resolution, dtype=torch.float32, minmax_scheme='999'):
    """Load data from an nc file.

    Args:
        dtype (torch.dtype, optional): Data type. Defaults to torch.float32.

    Returns:
        tuple[torch.Dataset, np.array]: Dataset containing train images and labels, 
        and the NumPy array containing test data.
    """

    #### load data
    test_pair = mlp.test_pair
    train, _ = get_data(test_pair, resolution=resolution, labeled=True, 
                           minmax_scheme=minmax_scheme)  
    train_images, train_labels = train

    device = set_device()
    
    dataset = TensorDataset(torch.tensor(train_images, device=device, dtype=dtype),
                            torch.tensor(train_labels, device=device, dtype=dtype))

    return dataset


def load_val_image(val_pair:tuple, resolution, hyperparameters):
    _, val = get_data(val_pair, resolution=resolution, labeled=True,
                minmax_scheme=hyperparameters['minmax_scheme'], silent=True)
    val_image, val_label = val

    return (val_image, val_label)


def train(models, trainloader, hyperparameters):
    mlp, autoencoder = models
    mlp.train()  # enable training

    encodedx, encodedy, encodedz = autoencoder.encoded_size
    resolution = autoencoder.in_resolution

    #### train MLP ####
    epochs = hyperparameters['epochs']
    optimizer = hyperparameters['optimizer']
    criterion = hyperparameters['criterion']

    loop = tqdm(range(epochs), desc='Training...', unit='epoch', colour='#7dc4e4')
    for epoch in loop:
        for _, batch_data in enumerate(trainloader):
            image, labels = batch_data  # feed in images and labels (V, P)

            optimizer.zero_grad()  # reset gradients

            encoding = mlp(labels).reshape(1, encodedx, encodedy, encodedz)  # forward pass, get mlp prediction from (v, p) then reshape
            output = autoencoder.decoder(encoding)  # get output image from decoder
            output = torchvision.transforms.functional.crop(output, 0, 0, resolution, resolution)  # crop to match shape

            loss = criterion(output, image)  # get the loss between input image and model output
            loss.backward()  # backward propagation
            optimizer.step()  # apply changes to network

            # print statistics
            loop.set_description(f"Epoch {epoch+1}/{epochs}")

    return None

def test(models, val_data) -> float:
    device = set_device()

    mlp, autoencoder = models
    mlp.eval()

    val_image, val_label = val_data
    val_label_T = torch.tensor(val_label, device=device, dtype=torch.float32)

    encodedx, encodedy, encodedz = autoencoder.encoded_size
    resolution = autoencoder.in_resolution

    # construct a fake encoding from a pair of (V, P) and reshape to the desired dimensions
    with torch.no_grad():
        # get MLP result on input pair
        fake_encoding = mlp(val_label_T)
        fake_encoding = fake_encoding.reshape(1, encodedx, encodedy, encodedz)  # reshape encoding from (1, xyz) to (1, x, y, z)
        decoded = autoencoder.decoder(fake_encoding)  # get image from decoder
        decoded = torchvision.transforms.functional.crop(decoded, 0, 0, resolution, resolution)  # safety measure: crop to required size

    decoded = decoded.cpu().numpy()
    # compute score (MSE)
    score = np.sqrt((decoded - val_image)**2).mean()

    return score


def reset_weights(model):
    if isinstance(model, nn.Linear):
        torch.nn.init.xavier_uniform_(model.weight.data)


def main(voltages, pressures):
    val_pairs = list(product(voltages, pressures))  # create list of cartesian products

    device = set_device()

    # read input from input folder
    root = Path.cwd()
    input_file = get_input_file(root)
    parameters, hyperparameters = read_input(input_file)
    torch.manual_seed(parameters['random_seed'])

    # load models 
    mlp, autoencoder = load_models(parameters)
    mlp.to(device)
    autoencoder.to(device)
    resolution = autoencoder.in_resolution

    now = dt.now()
    nowstring = now.strftime("%d-%m-%Y-%H%M")

    # set output
    out_dir = root/'k-fold'/nowstring
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
    shutil.copy(str(input_file), str(out_dir/'input.yml'))  # make a copy of the input file

    # load hyperparameters 
    hyperparameters = set_hyperparameters(mlp, hyperparameters)  # load criterion and optimizer properly

    # load train, test data
    dataset = load_data(mlp, resolution=resolution, 
                        minmax_scheme=hyperparameters['minmax_scheme'])
    trainloader = DataLoader(dataset, batch_size=1, shuffle=True)

    k = len(val_pairs)
    print(f'{k}-fold validation of ImageMLP {mlp.name}')
    scores = []

    #### begin loop
    for i, val_pair in enumerate(val_pairs):
        print(f'Training for pair {val_pair}, i={i}/{k}', end='\n')
        # get validation data
        val_data = load_val_image(val_pair, resolution, hyperparameters)

        #### train model 
        mlp.train()
        train((mlp, autoencoder), trainloader, hyperparameters)

        score = test((mlp, autoencoder), val_data)
        scores.append(score)

        # reset weights (same as starting a new model)
        mlp.apply(reset_weights)

    scores = np.array(scores)  # convert to array
    plot(scores, out_dir=out_dir)
    np.save(out_dir/'scores.npy', scores)

    write_metadata(mlp, out_dir)

    return scores  # MSE


def plot(scores:np.array, out_dir=None):
    from matplotlib.ticker import MultipleLocator, LogLocator
    fig, ax = plt.subplots(dpi=300)
    k = len(scores)

    mean = scores.mean()
    ax.set_yscale('log')
    ax.grid(axis='y',alpha=0.35, which='both', zorder=1)

    x = np.arange(k) + 1
    bars = ax.bar(x, scores, zorder=2)
    ax.bar_label(bars, fmt='%.3e', padding=3, size='small', color='tab:blue')

    ax.axhline(y=mean, c='k', ls=':')
    ax.annotate(f'$\mu = $ {mean:.2e}', xy=(0.2*k, mean), 
                xytext=(0, 5), textcoords='offset points')

    ax.xaxis.set_major_locator(MultipleLocator(1))

    ax.set_xlabel('k-th fold')
    ax.set_ylabel('Mean square error')

    if out_dir is not None:
        fig.savefig(out_dir/'plot.png', bbox_inches='tight')

    return None

if __name__ == '__main__':
    voltages = [300, 400]
    pressures = [30, 45, 60, 80, 100]

    now = dt.now()
    nowstring = now.strftime("%d-%m-%y at %H:%M:%S")
    print(f"Started on {nowstring}")
    
    main(voltages, pressures)

    now = dt.now()
    nowstring = now.strftime("%d-%m-%y at %H:%M:%S")
    print(f"k-fold validation finished on {nowstring}")
