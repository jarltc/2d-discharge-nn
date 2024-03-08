from plot import ae_correlation, plot_comparison_ae
from image_data_helpers import get_data, nc_data
from data_helpers import set_device
from autoencoder_classes import A64_6, A300
from mlp_classes import MLP

import torch
import torchvision
import xarray as xr
from pathlib import Path


def label_minmax(V, P):
    ds = xr.open_dataset(nc_data)
    v_list = list(ds.V.values)
    p_list = list(ds.P.values)

    vmax = max(v_list)
    vmin = min(v_list)
    pmax = max(p_list)
    pmin = min(p_list)

    sV = (V - vmin)/(vmax - vmin)
    sP = (P - pmin)/(pmax - pmin)

    return sV, sP


if __name__ == '__main__':
    device = set_device()
    root = Path.cwd()
    
    resolution = 64
    if resolution == 64:
        ae_model = A64_6()
        ae_model.load_state_dict(torch.load(ae_model.path))
        ae_model.to(device)

        encodedx = 40
        encodedy = 8
        encodedz = 8
        encoded_size = encodedx*encodedy*encodedz 

        out_dir = root/'created_models'/'conditional_autoencoder'/'64x64'/'A64g'
        mlp = MLP(2, encoded_size, dropout_prob=0.5)
        mlp.load_state_dict(torch.load(out_dir/'A64g'))
        mlp.to(device)

    elif resolution == 32:
        ae_model = A300()
        ae_model.load_state_dict(torch.load(ae_model.path))
        ae_model.to(device)

        encodedx = 20
        encodedy = 4
        encodedz = 4
        encoded_size = encodedx*encodedy*encodedz 

        mlp = MLP(2, encoded_size, dropout_prob=0.5)
        mlp.load_state_dict(torch.load(mlp.path32))
        mlp.to(device)
        out_dir = root/'created_models'/'conditional_autoencoder'/'32x32'/'A32g'


    # get scaled image test data
    _, test_data = get_data((300, 60), resolution=resolution, square=True)
    sV, sP = label_minmax(300, 60)
    labels = torch.tensor([sV, sP], device=device, dtype=torch.float32)

    mlp.eval()
    ae_model.eval()
    with torch.no_grad():
        fake_encoding = mlp(labels).reshape(1, encodedx, encodedy, encodedz)
        decoded = ae_model.decoder(fake_encoding)
        decoded = torchvision.transforms.functional.crop(
             decoded, 0, 0, resolution, resolution)

    r2 = ae_correlation(test_data, decoded, out_dir=out_dir)  # something wrong with this??
    plot_comparison_ae(test_data, fake_encoding, ae_model, 
                       out_dir=out_dir, is_square=True, resolution=resolution)
