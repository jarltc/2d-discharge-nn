""" Create image datasets for each pair of V, P

created by: jarl (28 Jul 2023)

xarray is nice, but I have no way of easily selecting train and test sets."""

import xarray as xr
import numpy as np
from pathlib import Path
from numpy import savez_compressed
from datetime import datetime as dt
from tqdm import tqdm

if __name__ == '__main__':
    # get list of v and p
    voltages = [200., 300., 400., 500.]
    pressures = [   5.,  10.,  30.,  45.,  60.,  80., 100., 120.]
    now = dt.now().strftime('%Y-%m-%d %H:%M')

    root = Path.cwd()/'data'
    data = root/'interpolation_datasets'/'full_interpolation.nc'
    ds = xr.open_dataset(data)
    variables = list(ds.keys())

    out_dir = root/'image_datasets'
    if not out_dir.exists():
        out_dir.mkdir()

    for p in tqdm(pressures):
        for v in voltages:
            array = np.nan_to_num(np.stack([ds[var].sel(V=v, P=p).values for var in variables]))
            v_name = f'{v:0>3}'
            p_name = f'{p:0>3}'
            filename = f'{v_name}_{p_name}.npz'  # format is VVV_PPP.npz
            savez_compressed(out_dir/filename, array)

    with open(out_dir/f'metadata.txt', 'w') as f:
        f.write(f'Created on {now}' + '\n')
        f.write(f'V = {voltages}' + '\n')
        f.write(f'P = {pressures}' + '\n')
