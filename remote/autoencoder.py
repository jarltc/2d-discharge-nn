"""Autoencoder for predicting (square cropped) 2d plasma profiles

Run this with the terminal (and quit VSCode) as much as possible to avoid excessive RAM usage.
"""

import time
from pathlib import Path
import yaml

import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
import torch.utils.benchmark as benchmark
from torchvision.transforms.functional import crop

import autoencoder_classes as AE
from plot import image_compare, save_history_graph, ae_correlation
from image_data_helpers import get_data, AugmentationDataset, load_synthetic, check_empty
from data_helpers import mse, set_device

def plot_train_loss(losses, validation_losses=None, out_dir=None):  # TODO: move to plot module

    losses = np.array(losses)
    fig, ax = plt.subplots(dpi=300)
    ax.set_yscale('log')
    ax.plot(losses, c='r', label='train')

    if validation_losses is not None:
        ax.plot(validation_losses, c='r', ls=':', label='validation')
        ax.legend()

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid()

    if out_dir is not None:
        fig.savefig(out_dir/'train_loss.png')


def write_metadata(model:nn.Module, hyperparameters:dict, times:dict, out_dir:Path):  # TODO: move to data modules
    # in_size = batch_data[0].size()  # broken
    in_resolution = model.in_resolution
    name = model.name
    epochs = hyperparameters["epochs"]
    learning_rate = hyperparameters["lr"]
    batch_size = hyperparameters["batch_size"]
    seed = hyperparameters["seed"]

    eval_time = times["eval"]
    train_time = times["train"]

    if in_resolution is None:
        in_size = (1, 5, 707, 200)  # TODO: stopgap measure
    else:
        in_size = (1, 5, in_resolution, in_resolution)

    # save model structure
    file = out_dir/'train_log.txt'
    with open(file, 'w') as f:
        f.write(f'Model {name}\n')
        f.write('***** layer behavior *****\n')
        print(summary(model, input_size=in_size, device='cuda'), file=f)
        print("\n", file=f)
        f.write('***** autoencoder architecture *****\n')
        print(model, file=f)
        f.write(f"\nSeed: {seed}")
        f.write(f'Epochs: {epochs}\n')
        f.write(f'Learning rate: {learning_rate}\n')
        f.write(f'Batch size: {batch_size}\n')
        f.write(f'Resolution: {in_resolution}\n')
        f.write(f'Evaluation time (100 trials): {eval_time.median * 1e3} ms\n')  # convert seconds to ms
        f.write(f'Train time: {train_time:.2f} seconds ({train_time/60:.2f} minutes)\n')
        f.write('\n***** end of file *****')
        

def speedtest(image: torch.tensor, model: nn.Module):
    """Evaluate prediction speeds for the model.

    Args:
        test_label (torch.tensor): Input pair of V, P from which predictions are made.

    Returns:
        torch.utils.benchmark.Measurement: The result of a Timer measurement. 
            Value can be extracted by the .median property.
    """
    timer = benchmark.Timer(stmt="autoencoder_eval(image, autoencoder)",
                            setup='from __main__ import autoencoder_eval',
                            globals={'image':image, 'autoencoder':model})

    return timer.timeit(100)


def autoencoder_eval(image, autoencoder):
    with torch.no_grad():
        output = autoencoder(image)
    return output


def scores(reference, prediction):
    # record scores
    return [mse(reference[0, i, :, :], prediction[0, i, :, :]) for i in range(5)]


def data_loading(model, dtype=torch.float32, minmax='999'):
    """Load data for training

    Args:
        model (nn.Module): Model class contains information for what kind of data it takes.
        dtype (torch.dtype, optional): Data type. Defaults to torch.float32.
        minmax (str): Minmax scheme. Defaults to 999 (99.9th percentile).

    Returns:
        train: Train data.
        test: Test data.
        val: Validation data.
    """
    in_resolution = model.in_resolution
    # resolution = (in_resolution, in_resolution)  # not used
    test_pair = model.test_pair
    val_pair = model.val_pair
    is_square = model.is_square
    ncfile = model.ncfile
    device = set_device()

    _, test, val = get_data(test_pair, val_pair, in_resolution, square=is_square, 
                            minmax_scheme=minmax)
    
    train = load_synthetic(ncfile, device, dtype)

    return train, test, val


def get_input_file(root):
    input_dir = root/'inputs'/'autoencoder'
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


def set_hyperparameters(input_file):
    
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

    yml = from_yaml(input_file)
    model = AE.get_model(yml['model'])
    learning_rate = yml['learning_rate']
    
    if yml['criterion'] == "MSE":
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"{yml['criterion']} is not recognized")

    if yml['optimizer'] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"{yml['optimizer']} is not recognized")
    

    hyperparameters = {"model": model,
                       "epochs": yml['epochs'],
                       "lr": learning_rate,
                       "criterion": criterion,
                       "optimizer": optimizer,
                       "batch_size": yml['batch_size'],
                       "seed" : yml['random_seed']}
    
    return hyperparameters


def training(model:nn.Module, data:tuple[torch.Tensor], hyperparameters:dict):
    """ Train the model """

    epochs = hyperparameters["epochs"]
    criterion = hyperparameters["criterion"]
    optimizer = hyperparameters["optimizer"]
    batch_size = hyperparameters["batch_size"]

    train, val = data
    trainloader = DataLoader(train, batch_size=batch_size, shuffle=True)

    loop = tqdm(range(epochs), desc='Training...', unit='epoch', colour='#7dc4e4')

    epoch_loss = []
    epoch_validation = []

    train_start = time.perf_counter()
    for epoch in loop:
        for _, batch_data in enumerate(trainloader):
            # get inputs
            inputs = batch_data  # TODO: this used to be batch_data[0] when using TensorDataset()
            optimizer.zero_grad()
            # record loss
            running_loss = 0.0

            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_description(f"Epoch {epoch+1}/{epochs}")

        with torch.no_grad():
            val_loss = criterion(model(val), val).item()

        epoch_validation.append(val_loss)
        epoch_loss.append(running_loss)
    
    train_end = time.perf_counter()

    return [(train_end-train_start), epoch_validation, epoch_loss]


def testing(model, test_tensor):
    """ Evaluate model performance """

    in_resolution = model.in_resolution

    if in_resolution is None:  # stopgap solution
        decoded = autoencoder_eval(test_tensor, model).cpu().numpy()  # no need to crop, handled by last layer
    else:
        decoded = autoencoder_eval(test_tensor, model).cpu().numpy()[:, :, :in_resolution, :in_resolution]  # convert to np?
    
    eval_time = speedtest(test_tensor, model)

    is_square = model.is_square
    test = test_tensor.cpu().numpy()
    out_dir = model.path

    fig = image_compare(test, decoded, out_dir, is_square, cmap='viridis')

    mse_scores = scores(test, decoded)  # TODO
    r2 = ae_correlation(test, decoded, out_dir)  # TODO

    return decoded, eval_time


def main(input_file):
    """Train model with a specific seed.

    Args:
        input_file (Path): Path to input file (yml) containing parameters for training a model.

    Returns:
        bool: Boolean if the predicted dataset contains an empty profile.
    """
    
    hyperparameters = set_hyperparameters(input_file)
    torch.manual_seed(hyperparameters['seed'])
    
    # set device
    device = set_device()
    dtype=torch.float32

    model = hyperparameters['model'].to(device)
    out_dir = model.path

    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    train, test, val = data_loading(model)
    val_tensor = torch.tensor(val, device=device, dtype=dtype)
    test_tensor = torch.tensor(test, device=device, dtype=dtype)

    data = (train, val_tensor)

    train_time, epoch_validation, epoch_loss = training(model, data, hyperparameters)

    decoded, eval_time = testing(model, test_tensor)

    times = {"train": train_time,
             "eval": eval_time}

    torch.save(model.state_dict(), out_dir/'model.pt')
    np.save(out_dir/'prediction.npy', decoded)
    plot_train_loss(epoch_loss,epoch_validation, out_dir)

    write_metadata(model, hyperparameters, times, out_dir)

    # check if the results are empty
    return check_empty(decoded)


if __name__ == '__main__':
    root = Path.cwd()
    in_file = get_input_file(root)
    result = main(in_file)
