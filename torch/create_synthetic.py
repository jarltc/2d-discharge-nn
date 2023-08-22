"""  Create synthetic images used to train the autoencoder. Images will be synthesized by different operations between data sets.

created by jarl on 22 Aug 2023
"""

from pathlib import Path
import itertools
import concurrent.futures
import time

import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

from image_data_helpers import get_dataset

# list of transforms
# average of any two datasets
# vertical flip
# horizontal flip

def average_pair(a, b, ds):
    """Process the average array between two pairs a and b.

    Args:
        a (tuple): Pair of (V, P)
        b (tuple): Pair of (V, P)

    Returns:
        _type_: _description_
    """
    variable_names = list(ds.keys())
    
    y = ds.y.values
    x = ds.x.values

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


if __name__ == "__main__":
    ncfile = Path('/Users/jarl/2d-discharge-nn/data/interpolation_datasets/rec-interpolation2.nc')
    ds = xr.open_dataset(ncfile)

    v_list = list(ds.V.values)
    p_list = list(ds.P.values)
    variable_names = list(ds.keys())

    vps = [(v, p) for v in v_list for p in p_list]  # get all pairs of V and P

    # add a combination of ((v1, p1), (v2, p2)) to the input queue
    start_time = time.perf_counter()
    pairs = list(itertools.combinations(vps, 2))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # submit tasks to the executor
        future_to_task = {executor.submit(average_pair, pair[0], pair[1], ds): pair for pair in pairs}

        # wait for tasks to complete and retrieve results
        results = []
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            results.append(future.result())

    # combine images (xr.datasets) into one big dataset
    averaged_dataset = xr.concat(results, dim='image')
    end_time = time.perf_counter()

    print(f'Processing took {end_time-start_time} s with multithreading\n')

    # write to file
    out_dir = ncfile.parent
    name = 'synthetic_averaged'
    out_file = out_dir/f'{name}.nc'
    write_job = averaged_dataset.to_netcdf(out_file, compute=False)

    with ProgressBar():
        print(f'Writing to {out_file}')
        write_job.compute()
