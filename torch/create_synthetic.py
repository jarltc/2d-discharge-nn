"""  Create synthetic images used to train the autoencoder. Images will be synthesized by different operations between data sets.

created by jarl on 22 Aug 2023
"""

import numpy as np
from pathlib import Path
import xarray as xr
from image_data_helpers import get_dataset
import itertools
import multiprocessing as mp

# list of transforms
# average of any two datasets
# vertical flip
# horizontal flip

def average_pair(a, b):
    """Process the average array between two pairs a and b.

    Args:
        a (tuple): Pair of (V, P)
        b (tuple): Pair of (V, P)

    Returns:
        _type_: _description_
    """
    global variable_names
    global y 
    global x

    va, pa = a
    vb, pb = b

    dataA = get_dataset(va, pa)
    dataB = get_dataset(vb, pb)

    # dataA and B have shapes of (channels, height, width), we need to get the 
    # average between the two that maintains the same shape
    z = (dataA + dataB) / 2.0  # numpy arrays (chanels, height, width)
    
    # process into xarray dataset
    channel_data_arrays = [xr.DataArray(z[i], dims=['y', 'x'],
                                        coords={'y':y, 'x':x}, 
                                        name=variable_names[i]) 
                                        for i in range(z.shape[0])]
    
    return xr.merge(channel_data_arrays)

ncfile = Path('/Users/jarl/2d-discharge-nn/data/interpolation_datasets/rec-interpolation2.nc')
ds = xr.open_dataset(ncfile)

v_list = list(ds.V.values)
p_list = list(ds.P.values)
variable_names = list(ds.keys())
x = ds.x.values
y = ds.y.values

vps = [(v, p) for v in v_list for p in p_list]  # get all pairs of V and P

# create a list of combinations of pairs of v, p = ((v1, p1), (v2, p2))
averaged_dataset = xr.concat([average_pair(a,b) for a, b in list(itertools.combinations(vps, 2))],
                             dim='image')

# write to file
out_dir = ncfile.parent
name = 'synthetic_averaged'
out_file = out_dir/f'{name}.nc'
write_job = ds.to_netcdf(out_file, compute=False)
from dask.diagnostics import ProgressBar
with ProgressBar():
    print(f'Writing to {out_file}')
    write_job.compute()
