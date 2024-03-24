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
from image_data_helpers import get_data, AugmentationDataset, load_synthetic
from data_helpers import mse, set_device

torch.manual_seed(28923)

def plot_train_loss(losses, validation_losses=None):  # TODO: move to plot module

    losses = np.array(losses)
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.plot(losses, c='r', label='train')

    if validation_losses is not None:
        ax.plot(validation_losses, c='r', ls=':', label='validation')
        ax.legend()

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid()

    fig.savefig(out_dir/'train_loss.png')


def write_metadata(out_dir):  # TODO: move to data modules
    # in_size = batch_data[0].size()  # broken
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
        f.write(f'Resolution: {resolution}\n')
        f.write(f'Evaluation time (100 trials): {eval_time.median * 1e3} ms\n')  # convert seconds to ms
        f.write(f'Train time: {(train_end-train_start):.2f} seconds ({(train_end-train_start)/60:.2f} minutes)\n')
        f.write('\n***** end of file *****')


class SimDataset(Dataset):
    def __init__(self, image_set, device, square=True):
        super().__init__()
        self.is_square = square
        self.device = device
        self.data = image_set  # ndarray: (channels, n_samples, height, width)

    def __len__(self):
        return self.data.shape[0]  # get number of V, P pairs
    
    def __getitem__(self, index):
        return torch.tensor(self.data[index], 
                            dtype=torch.float32, 
                            device=self.device)
        

def speedtest(image: torch.tensor):
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

if __name__ == '__main__':
    # set device
    device = set_device()

    model = AE.A64_9().to(device)
    name = model.name
    root = Path.cwd()

    out_dir = model.path
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
    dtype = torch.float32

    in_resolution = model.in_resolution
    resolution = (in_resolution, in_resolution)  # use a square image 
    test_pair = model.test_pair
    val_pair = model.val_pair
    is_square = model.is_square
    ncfile = model.ncfile

    _, test, val = get_data(test_pair, val_pair, in_resolution, square=is_square, 
                            minmax_scheme='999')

    train = load_synthetic(ncfile, device, dtype)
    trainloader = DataLoader(train, batch_size=32, shuffle=True)
    val_tensor = torch.tensor(val, device=device, dtype=dtype)

    ##### hyperparameters #####
    epochs = 500
    learning_rate = 1e-3

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    epoch_loss = []
    epoch_validation = []
    loop = tqdm(range(epochs), desc='Training...', unit='epoch', colour='#7dc4e4')

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
            val_loss = criterion(model(val_tensor), val_tensor).item()

        epoch_validation.append(val_loss)
        epoch_loss.append(running_loss)


    train_end = time.perf_counter()
    test_image = torch.tensor(test, device=device, dtype=dtype)

    decoded = autoencoder_eval(test_image, model).cpu().numpy()[:, :, :in_resolution, :in_resolution]  # convert to np?

    torch.save(model.state_dict(), out_dir/'model.pt')
    np.save(out_dir/'prediction.npy', decoded)
    eval_time = speedtest(test_image)
    fig = image_compare(test, decoded, out_dir, is_square, cmap='viridis')
    mse_scores = scores(test, decoded)
    r2 = ae_correlation(test, decoded, out_dir)
    plot_train_loss(epoch_loss, epoch_validation)
    write_metadata(out_dir)
