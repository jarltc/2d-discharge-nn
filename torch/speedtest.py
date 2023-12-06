# load model
import time
import torch
from pathlib import Path
from image_data_helpers import get_data
from autoencoder_classes import FullAE1, A64_8

def speedtest(resolution=None):
    """Get a trained model and evaluate its reconstruction speed.

    Args:
        resolution (int, optional): Resolution of the image. Defaults to None (full image).

    Returns:
        float: Time to reconstruct an image in ms.
    """
    device = torch.device("mps")
    if resolution == 64:
        square = True
        model_dir = Path("/Users/jarl/2d-discharge-nn/created_models/autoencoder/64x64/A64-8/A64-8")
        model = A64_8() 
    else:
        square = False
        model_dir = Path("/Users/jarl/2d-discharge-nn/created_models/autoencoder/fullAE-1/100ep_MSE/fullAE-1")
        model = FullAE1()

    _, test = get_data((300, 60), square=square, resolution=resolution)

    # convert to tensor
    image = torch.tensor(test, dtype=torch.float32, device=device)

    # load model
    model.load_state_dict(torch.load(model_dir))
    model.to(device)
    model.eval()

    start = time.perf_counter_ns()
    model(image)
    end = time.perf_counter_ns()

    return (end-start)/1e6

def regr(in_pair: tuple, resolution=None):
    """Get regression image for a given pair.

    I wrote this cause I got lazy to modify the existing cae_regr to accommodate full images.
    Uses plot_comparison_ae in plot.py.

    Args:
        in_pair (tuple): Input pair of V and P (not scaled).
        resolution (int, optional): Image resolution. Defaults to None (full image).
    """
    from plot import plot_comparison_ae

    device = torch.device("mps")
    if resolution == 64:
        square = True
        model_dir = Path("/Users/jarl/2d-discharge-nn/created_models/autoencoder/64x64/A64-8/A64-8")
        model = A64_8() 
    else:
        square = False
        model_dir = Path("/Users/jarl/2d-discharge-nn/created_models/autoencoder/fullAE-1/100ep_MSE/fullAE-1")
        model = FullAE1()

    _, test = get_data(in_pair, square=square, resolution=resolution)

    # convert to tensor
    sim = torch.tensor(test, dtype=torch.float32, device=device)

    # load model
    model.load_state_dict(torch.load(model_dir))
    model.to(device)
    model.eval()

    prediction = model.encoder(sim)
    out_dir = model_dir.parent

    plot_comparison_ae(test, prediction, model, out_dir=out_dir, is_square=square)
    print(f"file saved in {out_dir}")


if __name__ == "__main__":
    # full_time = speedtest()
    # small_time = speedtest(64)
     
    # return decode speed
    # print(f"Full AE speed: {full_time} ms")
    # print(f"64x64 speed: {small_time} ms")
    regr((500, 80))  # this shows metastable density more clearly
