# regr code for fullAE (autoencoder)
import time
import torch
import numpy as np
from pathlib import Path
from image_data_helpers import get_data
from autoencoder_classes import FullAE1, A64_8
import torch.utils.benchmark as benchmark

def speedtest(model_dir, resolution=None):
    """Get a trained model and evaluate its reconstruction speed.

    Args:
        resolution (int, optional): Resolution of the image. Defaults to None (full image).

    Returns:
        float: Time to reconstruct an image in ms.
    """
    device = torch.device("mps")
    if resolution == 64:
        square = True
        model_dir = ae_dir/'64x64'/'A64-8'/'A64-8'
        model = A64_8() 
    else:
        square = False
        model = FullAE1()

    # load test image
    _, test = get_data((300, 60), square=square, resolution=resolution)

    # convert to tensor
    image = torch.tensor(test, dtype=torch.float32, device=device)

    # load model
    model.load_state_dict(torch.load(model_dir))
    model.to(device)
    model.eval()

    t = benchmark.Timer(stmt="model(image)",
                        globals={'model':model, 'image':image})

    return t.timeit(1000)

def regr(in_pair: tuple, resolution=None):
    """Get regression image for a given pair.

    I wrote this cause I got lazy to modify the existing ae_regr to accommodate full images.
    Uses sep_comparison_ae in plot.py.

    Args:
        in_pair (tuple): Input pair of V and P (not scaled).
        resolution (int, optional): Image resolution. Defaults to None (full image).
    """
    from plot import plot_comparison_ae, sep_comparison_ae, sep_comparison_ae_v2

    minmax_scheme = '999'  # default for now

    device = torch.device("mps")
    if resolution == 64:
        square = True
        model_dir = ae_dir/'64x64'/'A64-8'/'A64-8'
        model = A64_8() 
    else:
        square = False
        model_dir = ae_dir/'fullAE-1'/'95622_500ep'/'fullAE-1'
        model = FullAE1()

    _, test = get_data(in_pair, square=square, resolution=resolution, minmax_scheme=minmax_scheme)  # change minmax scheme 

    # convert to tensor
    sim = torch.tensor(test, dtype=torch.float32, device=device)

    # load model
    model.load_state_dict(torch.load(model_dir))
    model.to(device)
    model.eval()

    prediction = model.encoder(sim)
    out_dir = model_dir.parent

    # plot_comparison_ae(test, prediction, model, out_dir=out_dir, is_square=square)
    sep_comparison_ae_v2(test, prediction, model, out_dir=out_dir, is_square=square, cbar='viridis', minmax_scheme=minmax_scheme)
    print(f"file saved in {out_dir}")


if __name__ == "__main__":
    root = Path.cwd()
    ae_dir = root/'created_models'/'autoencoder'
    fullae_dir = ae_dir/'fullAE-1'/'95622_seedtest2'/'fullAE-1'
    a64_dir = ae_dir/'64x64'/'A64-8'/'A64-8'
    small_time = speedtest(a64_dir, 64)
     
    # return decode speed
    # print(f"Full AE speed: {full_time} ms")
    # print(f"Full AE: {full_time.median*1e3:.3f} ms")
    print(f"64x64: {small_time.median*1e3:.3f} ms")
    # regr((300, 60))  
