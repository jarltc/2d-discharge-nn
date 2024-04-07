"""
Conditional autoencoder to reproduce 2d plasma profiles from a pair of V, P

* Load encoder
* Load MLP architecture

"""

import shutil
import time
from pathlib import Path
import yaml

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


def resize(data: np.ndarray, scale=64) -> np.ndarray:
    """Resize square images to 64x64 resolution by downscaling.

    Args:
        data (np.ndarray): Input data.

    Returns:
        np.ndarray: Downscaled input data.
    """

    data = np.stack([cv2.resize((np.moveaxis(image, 0, -1)), (scale, scale)) for image in data]) 
    return np.moveaxis(data, -1, 1)  # revert initial moveaxis


def normalize_test(dataset:np.ndarray, scalers:dict):
    normalized_variables = []

    for i, var in enumerate(['pot', 'ne', 'ni', 'nm', 'te']):
        x = dataset[:, i, :, :]
        xMin, xMax = scalers[var]
        scaledx = (x-xMin) / (xMax-xMin)  # shape: (31, x, x)
        normalized_variables.append(scaledx)
    
    # shape: (5, 31, x, x)
    normalized_dataset = np.moveaxis(np.stack(normalized_variables), 0, 1)  # shape: (31, 5, x, x)
    return normalized_dataset

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
    parameters = {'model': data['model'],
                  'autoencoder': data['autoencoder'],
                  'seed': data['random_seed']}

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
    autoencoder = AE.get_model(parameters['autoencoder']).eval()  # load autoencoder in inference mode
    autoencoder.load_state_dict(torch.load(autoencoder.path/'model.pt'))  # load parameters

    input_size = 2  # model takes two input values
    output_size = autoencoder.encoded_size
    
    model = mlp_classes.get_model(parameters['model'], input_size, output_size)
    return model, autoencoder


def get_input_file(root):
    input_dir = root/'inputs'/'imagemlp'
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


def mlp_eval(test_pair, mlp):
    with torch.no_grad():
        output = mlp(test_pair)
    return output


def speedtest(test_pair: torch.tensor, mlp: nn.Module):
    """Evaluate prediction speeds for the model.

    Args:
        test_label (torch.tensor): Input pair of V, P from which predictions are made.

    Returns:
        torch.utils.benchmark.Measurement: The result of a Timer measurement. 
            Value can be extracted by the .median property.
    """
    timer = benchmark.Timer(stmt="mlp_eval(test_pair, mlp)",
                            setup='from __main__ import mlp_eval',
                            globals={'test_pair':test_pair, 'mlp':mlp})

    return timer.timeit(100)


def write_metadata(models, input_file, times, out_dir):  # TODO: move to data module
    mlp, autoencoder = models
    resolution = autoencoder.in_resolution
    in_size = (1, 5, resolution, resolution)  # TODO: not sure if this is correct
    parameters, _ = read_input(input_file)
    train_time = times['train']
    eval_time = times['eval']
    device_name = set_device(name=True)

    # save model structure
    file = out_dir/'train_log.txt'
    with open(file, 'w') as f:
        # model description
        f.write(f'Model {mlp.name}, using autoencoder: {autoencoder.name}\n')
        print(summary(mlp, input_size=(1, 2), device=device_name), file=f)
        print("\n", file=f)
        print(summary(autoencoder, input_size=in_size), file=f)
        print("\n", file=f)
        print(autoencoder, file=f)
        print("\n", file=f)

        # training details
        f.write(f'Resolution: {resolution}\n')
        f.write(f"\nSeed: {parameters['seed']}")
        f.write(f'Evaluation time (100 trials): {eval_time.median * 1e3} ms\n')  # convert seconds to ms
        f.write(f'Train time: {train_time:.2f} seconds ({train_time/60:.2f} minutes)\n')
        f.write('\n***** end of file *****')


# TODO: break everything down into functions and bundle into main function
if __name__ == '__main__':
    device = set_device()
    
    # get input from input folder
    root = Path.cwd()
    input_file = get_input_file(root)
    parameters, hyperparameters = read_input(input_file)

    # load models #
    autoencoder, mlp = load_models(parameters)
    mlp.to(device)
    autoencoder.to(device)
    
    # load hyperparameters #
    hyperparameters = set_hyperparameters(mlp, hyperparameters)  # load criterion and optimizer properly
    name = mlp.name
    resolution = mlp.in_resolution
    encodedx, encodedy, encodedz = autoencoder.encoded_size
    encoded_size = encodedx*encodedy*encodedz
    
    # ----- #
    test_pair = mlp.test_pair
    val_pair = mlp.val_pair
    train, test, val = get_data(test_pair, val_pair, resolution=resolution, labeled=True, 
                           minmax_scheme=hyperparameters['minmax_scheme'])  
    train_images, train_labels = train
    test_image, test_label = test
    val_image, val_label = val

    out_dir = mlp.path
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
    shutil.copy(str(input_file), str(out_dir/'input.yml'))  # make a copy of the input file

    #### load data
    dataset = TensorDataset(torch.tensor(train_images, device=device, dtype=torch.float32),
                            torch.tensor(train_labels, device=device, dtype=torch.float32))
    trainloader = DataLoader(dataset, batch_size=1, shuffle=True)
    val_tensor = torch.tensor(val_image, device=device, dtype=torch.float32)

    #### train MLP ####
    epochs = hyperparameters['epochs']
    optimizer = hyperparameters['optimizer']
    criterion = hyperparameters['criterion']

    # record loss
    epoch_loss = []

    train_start = time.time()
    loop = tqdm(range(epochs), desc='Training...', unit='epoch', colour='#7dc4e4')
    for epoch in loop:
        running_loss = 0.0  # record losses

        for i, batch_data in enumerate(trainloader):
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
            running_loss += loss.item()

        epoch_loss.append(loss.item())
        if (epoch+1) % 10 == 0:
            # save model every 10 epochs (so i dont lose all training progress in case i do something unwise)
            torch.save(autoencoder.state_dict(), out_dir/'model.pt')

    train_end = time.time()
    torch.save(mlp.state_dict(), out_dir/'model.pt')
    save_history_graph(epoch_loss, out_dir)

    #### begin evaluation
    mlp.eval()

    # construct a fake encoding from a pair of (V, P) and reshape to the desired dimensions
    with torch.no_grad():
        # get MLP result on input pair
        fake_encoding = mlp(torch.tensor(test_label, device=device, dtype=torch.float32))  # mps does not support float64
        fake_encoding = fake_encoding.reshape(1, encodedx, encodedy, encodedz)  # reshape encoding from (1, xyz) to (1, x, y, z)
        decoded = autoencoder.decoder(fake_encoding)  # get image from decoder
        decoded = torchvision.transforms.functional.crop(decoded, 0, 0, resolution, resolution)  # crop to required size

    # add resolution=64 for larger images
    # eval_time, scores = plot_comparison_ae(test_image, fake_encoding, autoencoder, 
    #                                        out_dir=out_dir, is_square=True)
    eval_time = speedtest(torch.tensor(test_label, device=device, dtype=torch.float32), mlp)
    
    times = {'train': train_end - train_start,
             'eval': eval_time}
    
    # save stuff
    prediction = decoded.cpu().numpy()
    np.save(out_dir/'prediction.npy', prediction)
    write_metadata([mlp, autoencoder], input_file, times, out_dir)
    ae_correlation(test_image, decoded, out_dir, minmax=False)
    image_compare(test_image, prediction, out_dir, is_square=autoencoder.is_square, cmap='viridis')
    # train2db(out_dir, name, epochs, image_ds.v_excluded, image_ds.p_excluded, resolution, typ='mlp')
