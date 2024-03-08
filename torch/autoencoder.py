"""Autoencoder for predicting (square cropped) 2d plasma profiles

Jupyter eats up massive RAM so I'm making a script to do my tests
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
from image_data_helpers import get_data, AugmentationDataset
from data_helpers import mse


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
    # set metal backend (apple socs)
    # in_resolution = int(input('Resolution: ') or "1000")
    in_resolution = 200
    if in_resolution == 1000:
        resolution = None  # (707, 200)
    else:
        resolution = (in_resolution, in_resolution)  # use a square image 

    # set device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.backends.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # name = input("Enter model name: ")
    name = 'test'
    root = Path.cwd()

    out_dir = root/'created_models'/'autoencoder'/f'{resolution[0]}x{resolution[0]}'/name
    # out_dir = root/'created_models'/'autoencoder'/'32x32'/name
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    test_pair = (300, 60)
    val_pair = (400, 45)
    is_square = True
    dtype = torch.float32

    # get augmentation data, this changes depending on the resolution
    synthetic_dir = Path('/Users/jarl/2d-discharge-nn/data/interpolation_datasets/synthetic')
    if resolution == 32:
        ncfile = synthetic_dir/'synthetic_averaged_s32.nc'
    elif resolution == 64:
        ncfile = synthetic_dir/'synthetic_averaged_s64.nc'
    elif resolution == 200:
        # TODO: create 200x200 dataset
        ncfile = synthetic_dir/'synthetic_averaged.nc'
    else:
        ncfile = Path('/Users/jarl/2d-discharge-nn/data/interpolation_datasets/synthetic/synthetic_averaged.nc')

    _, test, val = get_data(test_pair, val_pair, in_resolution, square=is_square)

    augdataset = AugmentationDataset(ncfile, device, is_square=is_square)
    trainloader = DataLoader(augdataset, batch_size=32, shuffle=True)
    val_tensor = torch.tensor(val, device=device, dtype=dtype)

    ##### hyperparameters #####
    epochs = 5
    learning_rate = 1e-3

    if in_resolution == 32:
        model = AE.A300().to(device)
    elif in_resolution == 64:
        model = AE.A64_9().to(device)
    elif in_resolution == 200:
        model = AE.A200_1().to(device)  # still missing
    else: model = AE.FullAE1().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    epoch_loss = []
    epoch_validation = []
    loop = tqdm(range(epochs), desc='Training...', unit='epoch', colour='#7dc4e4')

    # patience = 30
    # best_loss = 100
    # best_epoch = -1
    # eps = 1e-5  # threshold to consider if the change is significant
    # epochs_since_best = 0

    train_start = time.perf_counter()
    for epoch in loop:
        for i, batch_data in enumerate(trainloader):
            # get inputs
            inputs = batch_data  # TODO: this used to be batch_data[0] when using TensorDataset()
            optimizer.zero_grad()
            if in_resolution == 200:
                inputs = crop(inputs, 550, 0, 200, 200)  # image, top, left, height, width

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

        ### -----  EARLY STOPPPING ----- ###
        # check if the current loss is smaller than the best loss 
        # if (val_loss < best_loss):
        #     # update the best epoch and loss
        #     best_epoch = epoch  

        #     if (abs(val_loss - best_loss) < eps):
        #         # if change is insignificant, keep waiting
        #         epochs_since_best += 1

        #         if epochs_since_best >= patience:
        #             # save model when best result is reached and model has not improved a number of times
        #             epochs = best_epoch + 1
        #             torch.save(model.state_dict(), out_dir/f'{name}')
        #             save_history_graph(epoch_loss, out_dir)
        #             break
        #     else:
        #         # if loss significantly decreases, reset progress
        #         epochs_since_best = 0
            
        #     best_loss = val_loss  # update the loss after comparisons have been made


        if (epoch+1) % epochs == 0:
            # save model every 10 epochs (so i dont lose all training progress in case i do something unwise)
            torch.save(model.state_dict(), out_dir/f'{name}') 
            save_history_graph(epoch_loss, out_dir)

    train_end = time.perf_counter()
    test_image = torch.tensor(test, device=device, dtype=dtype)

    decoded = autoencoder_eval(test_image, model).cpu().numpy()[:, :, :in_resolution, :in_resolution]  # convert to np?

    torch.save(model.state_dict(), out_dir/f'{name}')
    np.save(out_dir/'prediction.npy', decoded)
    eval_time = speedtest(test_image)
    fig = image_compare(test, decoded, out_dir, is_square, cmap='viridis')
    mse_scores = scores(test, decoded)
    r2 = ae_correlation(test, decoded, out_dir)
    plot_train_loss(epoch_loss, epoch_validation)
    write_metadata(out_dir)
