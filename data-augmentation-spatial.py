#coding: utf-8
"""Grid interpolation for data augmentation

Create datasets of linear interpolation from the simulation data's
mesh grid to a fine linear grid. step controls the size of the grid points.
Data is imported from .dat files and output as a single NetCDF file containing
variable names and coordinate labels: https://docs.xarray.dev/en/stable/user-guide/data-structures.html.

@author: jarl
Created on Thu 15 Dec 2022
"""

import os
import pandas as pd
from pathlib import Path
import numpy as np
import numpy.ma as ma
import xarray as xr
from scipy import interpolate
from tqdm import tqdm

# borrowed from data.py
import re

def read_file(file_path):
    with open(file_path, 'r') as f:
        data = []
        for n,line in enumerate(f,1):
            if n==1:
                line = line.strip()
                line = re.findall(r'"[^"]*"', line) # get ['"var_name1"', '"var_name2"', ...]
                column_labels = [var_name.replace('"','') for var_name in line]
            elif n==2:
                continue
            else:
                data_line = [eval(data) for data in line.split()]
                if len(data_line)==4:
                    break
                data.append(data_line)
    return pd.DataFrame(data, columns=column_labels)


def create_mask(X, Y):
    # masks (in mm) to m
    # top electrode
    mask1 = ((X >= 0*1e-3) & (X <= 95*1e-3)) & ((Y >= 487*1e-3) & (Y <= 489*1e-3))
    mask2 = ((X >= 0*1e-3) & (X <= 40*1e-3)) & ((Y >= 453*1e-3) & (Y <= 487*1e-3))

    # bottom electrode
    mask3 = ((X >= 0*1e-3) & (X <=  95*1e-3)) & ((Y >= 395*1e-3) & (Y <= 415*1e-3))
    mask4 = ((X >= 0*1e-3) & (X <=  90*1e-3)) & ((Y >= 310*1e-3) & (Y <= 395*1e-3))
    mask5 = ((X >= 0*1e-3) & (X <= 120*1e-3)) & ((Y >= 277*1e-3) & (Y <= 310*1e-3))
    mask6 = ((X >= 0*1e-3) & (X <=  90*1e-3)) & ((Y >=   0*1e-3) & (Y <= 277*1e-3))

    # floating electrode
    mask7 = ((X >= 122*1e-3) & (X <= 185*1e-3)) & ((Y >= 224*1e-3) & (Y <= 234*1e-3))

    mask = (mask1 | mask2 | mask3 | mask4 | mask5 | mask6 | mask7)

    return mask


def interpolation(df, parameter, voltage, pressure):
    """ Perform interpolation for a parameter in a certain file.

    Args:
        df (DataFrame): DataFrame of a single dataset
        parameter (str): Column label in parameters (list)

    Returns:
        DataArray: DataArray of the interpolated data. Dimensions are V, P, x, y
    """
    # get points and values over which the interpolation is performed
    points = df[['X', 'Y']].to_numpy()
    values = df[parameter].to_numpy()
   
    # interpolation: griddata takes points([[x], [y]]) and values f(x,y)
    # to perform interpolation
    z = interpolate.griddata(points, values, (X, Y), method='linear')

    # convert to DataArray, z is in shape = (y, x) so label accordingly
    # array contains data for the parameter interpolated to a linear grid
    da = xr.DataArray(z, coords={'y':y, 'x':x}, dims=['y', 'x'], name=parameter)
    
    # assign coordinates V and P to be the voltage and pressure of the dataset, then expand dims
    return da.assign_coords(V=voltage, P=pressure).expand_dims(dim=['V','P'])


###### main #######
root = Path(os.getcwd())
data_folder = root/'data'/'avg_data'
out_dir = root/'data'/'interpolation_datasets'
if not os.path.exists(out_dir): os.mkdir(out_dir)

files = [file for file in data_folder.rglob('*')]

excluded = '300Vpp_060Pa_node'

# remove test data
for i, file in enumerate(files):
    if file.stem == excluded:
        files.pop(i)

step = 0.001 # meters

# list of parameters for the interpolation
parameters = ['potential (V)', 
            #   'Ex (V/m)', 
            #   'Ey (V/m)', 
              'Ne (#/m^-3)', 
              'Ar+ (#/m^-3)', 
              'Nm (#/m^-3)', 
              'Te (eV)']

# list of datasets to be concat at the end
ds_list = []
# process each file
for file in tqdm(files):
    # get voltage and pressure from filename
    v_string, p_string, _ = file.stem.split('_')
    voltage = float(v_string[:3])
    pressure = float(p_string[:3])

    # read file and get extents of spatial coordinates
    df = read_file(file)
    xmin = df['X'].min()
    xmax = df['X'].max()
    ymin = df['Y'].min()
    ymax = df['Y'].max()

    # original is in meters (why?), divide into 1mm x 1mm cells (step = 0.001)
    x = np.arange(xmin, xmax, step)
    y = np.arange(ymin, ymax, step)

    # create meshgrid as input for the masking and interpolation
    X, Y = np.meshgrid(x, y)
   
    # convert points in the electrodes to nan to avoid interpolation inside
    mask = create_mask(X, Y)
    X = ma.array(X, mask=mask).filled(fill_value=np.nan)
    Y = ma.array(Y, mask=mask).filled(fill_value=np.nan)

    # create a list of DataArrays for each variable in the file
    var_array = [interpolation(df, parameter, voltage, pressure) for parameter in parameters]
    # var_array = []
    # for parameter in parameters:
    #     points = df[['X', 'Y']].to_numpy()
    #     values = df[parameter].to_numpy()
    #     z = interpolate.griddata(points, values, (X, Y), method='linear')
    #     # z = ma.array(z, mask=mask).filled(fill_value=np.nan)
    #     da = xr.DataArray(z, coords={'y':y, 'x':x}, dims=['y', 'x'], name=parameter)
    #     var_array.append(da.assign_coords(V=voltage, P=pressure).expand_dims(dim=['V','P']))
    
    # assign coordinates V and P to be the voltage and pressure of the dataset, then expand dims
    # return da.assign_coords(V=voltage, P=pressure).expand_dims(dim=['V','P'])        
    ds_list.append(xr.merge(var_array))  # merge into Dataset

# create one large Dataset from ds_list and save
print('Concatenating datasets...')
ds = xr.combine_by_coords(ds_list)
name = 'rec-interpolation'
out_file = out_dir/'{name}.nc'

# write the file in chunks
write_job = ds.to_netcdf(out_file, compute=False)
from dask.diagnostics import ProgressBar
with ProgressBar():
    print(f"Writing to {out_file}")
    write_job.compute()

metadata = {'dataset excluded': excluded,
            'grid spacing (m)' : step,
            'masking' : 'before interpolation',
            'file size (mb)' : os.path.getsize(out_file) / 1e6,
            'sizes' : str(ds.sizes)}

with open(out_dir/'{name}_metadata.txt', 'w') as f:
    for key, value in metadata.items():
        f.write(key + ': ' + str(value) + '\n')