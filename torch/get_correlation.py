from plot import ae_correlation
from image_data_helpers import get_data, nc_data
from autoencoder_classes import A64_6
from mlp_classes import MLP

import torch
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
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    ae_model = A64_6()
    ae_dir = torch.load(ae_model.path)
    ae_model.load_state_dict(ae_dir)
    ae_model.to(device)

    encodedx = 40
    encodedy = 8
    encodedz = 8
    encoded_size = encodedx*encodedy*encodedz 
    mlp = MLP(2, encoded_size, dropout_prob=0.5)
    mlp_dir = torch.load(mlp.path64)
    mlp.load_state_dict(mlp_dir)
    mlp.to(device)

    # get scaled image test data
    _, test_data = get_data((300, 60), resolution=64, square=True)
    sV, sP = label_minmax(300, 60)
    labels = torch.tensor([sV, sP], device=device, dtype=torch.float32)

    # get predictions
    # not needed ?
    # test_data = torch.tensor(test_data, dtype=torch.float32, device=device)

    mlp.eval()
    ae_model.eval()
    with torch.no_grad:
        fake_encoding = mlp(labels)
        fake_encoding = fake_encoding.reshape(1, encodedx, encodedy, encodedz)
        decoded = ae_model.decoder(fake_encoding)

    out_dir = Path('/Users/jarl/2d-discharge-nn/created_models/conditional_autoencoder/64x64/A64g')
    r2 = ae_correlation(test_data, fake_encoding, out_dir)
