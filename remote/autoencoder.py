"""Autoencoder for predicting (square cropped) 2d plasma profiles

Run this with the terminal (and quit VSCode) as much as possible to avoid excessive RAM usage.
"""

import time
from pathlib import Path

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

    eval_time = times["eval"]
    train_time = times["train"]

    in_size = (1, 5, in_resolution, in_resolution)

    # save model structure
    file = out_dir/'train_log.txt'
    with open(file, 'w') as f:
        f.write(f'Model {name}\n')
        f.write('***** layer behavior *****\n')
        print(summary(model, input_size=in_size, device='mps'), file=f)
        print("\n", file=f)
        f.write('***** autoencoder architecture *****\n')
        print(model, file=f)
        f.write(f'\nEpochs: {epochs}\n')
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


def data_loading(model, dtype=torch.float32):
    in_resolution = model.in_resolution
    resolution = (in_resolution, in_resolution)  # use a square image 
    test_pair = model.test_pair
    val_pair = model.val_pair
    is_square = model.is_square
    ncfile = model.ncfile
    device = set_device()

    _, test, val = get_data(test_pair, val_pair, in_resolution, square=is_square, 
                            minmax_scheme='999')
    
    train = load_synthetic(ncfile, device, dtype)

    return train, test, val


def set_hyperparameters(model):
    epochs = 200
    learning_rate = 1e-3
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    batch_size = 128

    hyperparameters = {"epochs": epochs,
                       "lr": learning_rate,
                       "criterion": criterion,
                       "optimizer": optimizer,
                       "batch_size": batch_size}
    
    return hyperparameters


def training(model:nn.Module, data:tuple[torch.Tensor], hyperparameters:dict):

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
        for i, batch_data in enumerate(trainloader):
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

    in_resolution = model.in_resolution
    is_square = model.is_square
    device = set_device()
    test = test_tensor.cpu().numpy()
    out_dir = model.path

    decoded = autoencoder_eval(test_tensor, model).cpu().numpy()[:, :, :in_resolution, :in_resolution]  # convert to np?
    eval_time = speedtest(test_tensor, model)

    fig = image_compare(test, decoded, out_dir, is_square, cmap='viridis')

    mse_scores = scores(test, decoded)
    r2 = ae_correlation(test, decoded, out_dir)

    return decoded, eval_time


def main(seed):
    """Train model with a specific seed.

    Args:
        seed (int): Random number seed for PyTorch objects.

    Returns:
        bool: Boolean if the predicted dataset contains an empty profile.
    """
    torch.manual_seed(seed)
    
    # set device
    device = set_device()
    dtype=torch.float32

    model = AE.A64_9().to(device)
    out_dir = model.path

    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    train, test, val = data_loading(model)
    val_tensor = torch.tensor(val, device=device, dtype=dtype)
    test_tensor = torch.tensor(test, device=device, dtype=dtype)

    hyperparameters = set_hyperparameters(model)

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
    seed = 28923
    result = main(seed)
    print(result)
