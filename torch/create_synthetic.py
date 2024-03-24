"""  Create synthetic images used to train the autoencoder. Images will be synthesized by different operations between data sets.

Parallelism performed using concurrent.futures (https://docs.python.org/3/library/concurrent.futures.html)

The nc file created has 32C2 images (496) with 5 variables, and coordinates for y and x.

created by jarl on 22 Aug 2023
"""

from pathlib import Path
import itertools
import concurrent.futures
import time

import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

from image_data_helpers import get_dataset, crop, downscale, minmax_scale, get_maxima

# list of transforms
# average of any two datasets
# vertical flip
# horizontal flip

def average_pair(a, b, ds, resolution=None):
    """Process the average array between two pairs a and b.

    Args:
        a (tuple): Pair of (V, P)
        b (tuple): Pair of (V, P)

    Returns:
        _type_: _description_
    """
    variable_names = list(ds.keys())
    
    # convert coordinates based on scaling if resolution provided
    if resolution is not None:
        y = np.linspace(0, max(ds.y.values), resolution)
        x = np.linspace(0, max(ds.x.values), resolution)
    else:
        y = ds.y.values
        x = ds.x.values

    va, pa = a
    vb, pb = b

    dataA = get_dataset(va, pa)
    dataB = get_dataset(vb, pb)

    # dataA and B have shapes of (channels, height, width), we need to get the 
    # average between the two that maintains the same shape
    z = (dataA + dataB) / 2.0  # numpy arrays (chanels, height, width)

    # crop to square
    if resolution is not None:
        cropped = crop(z)
        # downscale 
        dscaled = downscale(cropped, resolution)
    else:
        dscaled = z
    maxima = get_maxima(ds, minmax_scheme='999')
    scaled = minmax_scale(dscaled, maxima)
    
    # process into xarray dataset
    channel_data_arrays = [xr.DataArray(scaled[i], dims=['y', 'x'],
                                        coords={'y':y, 'x':x}, 
                                        name=variable_names[i]) 
                                        for i in range(z.shape[0])]
    
    return xr.merge(channel_data_arrays)


if __name__ == "__main__":
    root = Path.cwd()
    ncfile = root/'data'/'interpolation_datasets'/'rec-interpolation2.nc'
    ds = xr.open_dataset(ncfile)

    v_list = list(ds.V.values)
    p_list = list(ds.P.values)
    variable_names = list(ds.keys())

    resolution = 64
    if resolution is not None:
        square = True
    suffix = f's{resolution}'

    vps = [(v, p) for v in v_list for p in p_list]  # get all pairs of V and P

    # add a combination of ((v1, p1), (v2, p2)) to the input queue
    print('job started')
    start_time = time.perf_counter()
    pairs = list(itertools.combinations(vps, 2))
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:  # Future objects represent the result of an asynchronous task.
        # submit tasks to the executor, performs average_pair() on a pair in the list initialized earlier.
        future_to_task = {executor.submit(average_pair, pair[0], pair[1], ds, resolution): pair for pair in pairs}  # creates a dictionary of future: task, where a task serves
                                                                                                        # as the key to a future.

        # wait for tasks to complete and retrieve results
        results = []
        for future in concurrent.futures.as_completed(future_to_task):  # iterates over Future objects as they complete. When a Future has finished, it yields that Future
            results.append(future.result())  # get the result associated with a Future

    # combine images (xr.datasets) into one big dataset
    averaged_dataset = xr.concat(results, dim='image')
    averaged_dataset.attrs['Description'] = "Dataset of minmax-scaled profiles (99.9th percentile) and downscaled to 64x64."
    end_time = time.perf_counter()

    print(f'Processing took {end_time-start_time} s with multiprocessing\n')

    # write to file
    out_dir = ncfile.parent
    name = 'synthetic_averaged999'
    if square:
        out_file = out_dir/f'{name}_s{resolution}.nc'
    else:
        out_file = out_dir/f'{name}.nc'
    write_job = averaged_dataset.to_netcdf(out_file, compute=False)

    with ProgressBar():
        print(f'Writing to {out_file}')
        write_job.compute()
