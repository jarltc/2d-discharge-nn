# load data and scale it according to a specified minmax scheme

import xarray as xr
import numpy as np
from pathlib import Path
from image_data_helpers import get_dataset
from dask.diagnostics import ProgressBar


# from image_data_helpers
def minmax_scale(image:np.ndarray, ds:xr.Dataset):
    """Perform minmax scaling on some input image with shape (channels, height, width)

    Values for the minmax scaling are obtained from the .nc file. 
    This also forces the minimum to be 0 instead of some crazy value that might mess with calculations.
    This scheme scales an array to lie between 0 and 1 (the maximum value for the specific variable across the entire dataset).
    Args:
        image (np.ndarray): Image (channels, height, width) to be scaled.
        ds (xr.Dataset): Dataset to reference for the maximum values of each parameter.

    Returns:
        np.ndarray: Minmax-scaled array.
    """

    scaled_arrays = []

    for i, variable in enumerate(list(ds.keys())):

        var_data = np.nan_to_num(ds[variable].values)
        a = 0.0  # force 0 minimum
        b = np.quantile(var_data, 0.999)

        data_array = image[i, :, :]

        # minmax scale
        scaled_arrays.append((data_array-a) / (b - a))

    scaled = np.stack(scaled_arrays)

    assert scaled.shape == image.shape

    return scaled


root = Path.cwd()
data_dir = root/'data'/'interpolation_datasets'/'synthetic'

ds = xr.open_dataset(data_dir/'synthetic_averaged.nc')
y = ds.y.values
x = ds.x.values

vars = list(ds.keys())
my_dict = my_dict = {var: None for var in vars}
var_arrays = []

for var in vars:
    data = ds[var].values
    q999 = np.quantile(data, 0.999)
    my_dict[var] = q999

a = 0.0  # min value for minmax scaling

var_arrays = [xr.DataArray((ds[var].values - a)/(my_dict[var] - a), 
                           dims=['image', 'y', 'x'],
                           coords={'image': range(496), 'y':y, 'x':x},
                           name=var) 
                           for var in vars]

ds99 = xr.merge(var_arrays)
out_file = data_dir/'synthetic_averaged999.nc'
write_job = ds99.to_netcdf(out_file, compute=False)
with ProgressBar():
    print(f'Writing to {out_file}')
    write_job.compute()
