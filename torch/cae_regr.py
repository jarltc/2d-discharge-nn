"""Autoencoder for predicting 2d plasma profiles

Jupyter eats up massive RAM so I'm making a script to do my tests
"""

import os
import time
import pickle
from pathlib import Path

import matplotlib.pyplot as plt

import cv2
import numpy as np

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import crop
import torch.utils.benchmark as benchmark

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from data_helpers import ImageDataset
from plot import plot_comparison_ae, save_history_graph, ae_correlation, image_slices, delta, image_compare
import autoencoder_classes
import mlp_classes
from image_data_helpers import get_data
from data_helpers import mse

# define model TODO: construct following input file/specification list

def resize(data: np.ndarray, scale=64) -> np.ndarray:
    """Resize square images to 64x64 resolution by downscaling.

    Args:
        data (np.ndarray): Input data.

    Returns:
        np.ndarray: Downscaled input data.
    """

    data = np.stack([cv2.resize((np.moveaxis(image, 0, -1)), (scale, scale)) for image in data])
    data = np.moveaxis(data, -1, 1)
    return data


def normalize_test(dataset:np.ndarray, scalers:dict()):
    normalized_variables = []

    for i, var in enumerate(['pot', 'ne', 'ni', 'nm', 'te']):
        x = dataset[:, i, :, :]  # plot the figures, vmax depends on which set (true or prediction) contains the higher values
        xMin, xMax = scalers[var]
        scaledx = (x-xMin) / (xMax-xMin)  # shape: (31, x, x)
        normalized_variables.append(scaledx)
    
    # shape: (5, 31, x, x)
    normalized_dataset = np.moveaxis(np.stack(normalized_variables), 0, 1)  # shape: (31, 5, x, x)
    return normalized_dataset


def write_metadata(out_dir):  # TODO: move to data module

    file = out_dir/'regr_metadata.txt'
    with open(file, 'w') as f:
        print(autoencoder, file=f)
        f.write(f'Resolution: {resolution}\n')
        f.write(f'Evaluation time: {eval_time:.2f} ms\n')
        f.write(f'Scores (MSE): {mse_scores}\n')
        f.write(f'Scores (r2): {r2}\n')
        f.write('\n***** end of file *****')


def imageMLP_eval(test_label, mlp, autoencoder):
    """Model evaluation for ImageMLP.

    Args:
        mlp (torch.nn.Module): ImageMLP.
        autoencoder (torch.nn.Module): Autoencoder to be used to produce the output image.
        test_label (_type_): _description_

    Returns:
        _type_: _description_
    """
    global encodedx, encodedy, encodedz

    with torch.no_grad():  # disable gradients computation
        fake_encoding = mlp(torch.tensor(test_label, device=device, dtype=torch.float32))\
            .reshape(1, encodedx, encodedy, encodedz)  # mps does not support float64
        decoded = autoencoder.decoder(fake_encoding).cpu().numpy()[:, :, :resolution, :resolution]

    return fake_encoding, decoded


def speedtest(test_label: torch.tensor):
    """Evaluate prediction speeds for the model.

    Args:
        test_label (torch.tensor): Input pair of V, P from which predictions are made.

    Returns:
        torch.utils.benchmark.Measurement: The result of a Timer measurement. 
            Value can be extracted by the .median property.
    """
    timer = benchmark.Timer(stmt="imageMLP_eval(test_label, mlp, autoencoder)",
                            setup='from __main__ import imageMLP_eval',
                            globals={'test_label':test_label, 'mlp':mlp, 'autoencoder':autoencoder})

    return timer.timeit(100)


def scores(reference, prediction):
    # record scores
    return [mse(reference[0, i, :, :], prediction[0, i, :, :]) for i in range(5)]


if __name__ == '__main__':
    # set metal backend (apple socs)
    device = torch.device(
        'mps' if torch.backends.mps.is_available() else 'cpu')

    root = Path.cwd()
    is_square=True

    resolution = 64
    if resolution == 32:  # (4, 4, 20)
        autoencoder = autoencoder_classes.A300()
    elif resolution == 64:  # (8, 8, 40)
        autoencoder = autoencoder_classes.A64_8()
    
    ae_dir = Path('/Users/jarl/2d-discharge-nn/created_models/autoencoder/64x64/A64-8/A64-8')
    mlp_dir = Path(input('Path to MLP: '))

    encodedx, encodedy, encodedz = autoencoder.encoded_shape
    encoded_size = encodedx * encodedy * encodedz

    mlp = mlp_classes.MLP4(2, encoded_size, dropout_prob=0.5)

    out_dir = mlp_dir.parents[0]
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    image_ds = ImageDataset(root/'data'/'interpolation_datasets', is_square)
    _, test = get_data((300, 60), resolution=resolution, square=is_square, labeled=True)
    test_image, test_label = test

    autoencoder.load_state_dict(torch.load(ae_dir))  # use path directly to model
    mlp.load_state_dict(torch.load(mlp_dir))  # use path directly to model
    
    # move models to gpu
    autoencoder.to(device)
    mlp.to(device)

    # place models in evaluation mode
    mlp.eval()
    autoencoder.eval()
    
    fake_encoding, prediction = imageMLP_eval(test_label, mlp, autoencoder)

    eval_time = speedtest(test_label).median * 1e3  # seconds to ms
    fig = image_compare(test_image, prediction, out_dir=out_dir, is_square=is_square, cmap='viridis', unscale=True)
    mse_scores = scores(test_image, prediction)
    r2 = ae_correlation(test_image, prediction, out_dir)
    # slices = image_slices(test_image, decoded, out_dir=out_dir, cmap='viridis')
    # delta_fig = delta(test_image, decoded, out_dir=out_dir, is_square=True)
    write_metadata(out_dir)
