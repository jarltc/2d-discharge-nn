''' Test code to process netcdf to tensorflow tensor'''

from pathlib import Path
import tensorflow as tf
import numpy as np
import xarray as xr
import pandas as pd
import os

import data

# check how previous tensors look like
datfile = Path('/Users/jarl/2d-discharge-nn/data/avg_data/200Vpp_045Pa_node.dat')
ds = xr.open_dataset(Path('/Users/jarl/2d-discharge-nn/data/interpolation_datasets/rec-interpolation.nc'))

