#coding: utf-8
"""
Data augmentation.

Create linear interpolation of data sets between V and P.
Ex and Ey in the files generated here are not valid.
Created on Mon Dec 5

@author: jarl
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tqdm import tqdm
from pathlib import Path
import data

def get_data_table(data_dir, voltage, pressure):
    file_name = '{0:d}Vpp_{1:03d}Pa_node.dat'.format(voltage,pressure)
    file_path = data_dir/ file_name
    if os.path.exists(file_path):
        avg_data = data.read_all_data(data_dir, [voltage], [pressure])
    else:
        avg_data = create_dummy_data_table(data_dir, voltage, pressure)
    return avg_data


def create_dummy_data_table(data_dir, voltage, pressure):
    ext_data = data.read_all_data(data_dir, [300], [60])
    ext_data = ext_data.drop(columns=['Vpp [V]','P [Pa]'])
    
    XY_df = ext_data.iloc[:,:2]
    ext_data = ext_data.drop(columns=['X','Y'])
    
    nan_mat = np.zeros_like(ext_data.values)
    nan_mat[:,:] = np.nan
    Nan_df = pd.DataFrame(nan_mat, columns= ext_data.columns)
    
    dummy_df = pd.concat([XY_df,Nan_df], axis=1)
    
    dummy_df = data.attach_VP_columns(dummy_df, voltage, pressure)
    
    return dummy_df


def interpolation(parameter, interpolation_variable=None):
    """ Interpolate between two arrays

    Args:
        parameter (str): Column name for parameter to interpolate.
        interpolation_variable (str, optional): Interpolate across voltage or pressure. Defaults to None.

    Raises:
        Exception: Raised when interpolation variable is not specified.

    Returns:
        pd.Series: pd.Series of new column created from interpolation.
    """
    if interpolation_variable == 'voltage':
        start_variable = start_voltage
        end_variable = end_voltage
        mid_variable = inter_voltage
    elif interpolation_variable == 'pressure':
        start_variable = start_pressure
        end_variable = end_pressure
        mid_variable = inter_pressure
    else: 
        raise Exception('Interpolation variable not specified!')

    start_array = start_df[parameter].values
    end_array = end_df[parameter].values
    interpolator = interp1d([start_variable, end_variable], np.vstack([start_array, end_array]), axis=0)
    inter_array = pd.Series(interpolator(mid_variable), name=parameter)
    
    return inter_array


data_folder = Path('/Users/jarl/2d-discharge-nn/data/avg_data')

voltages = [200, 300, 400, 500]
pressures = [5, 10, 30, 45, 60, 80, 100, 120]

# pair values with a moving window
v_pairs = [(a, b) for a, b in list(zip(voltages, voltages[1:]))]
p_pairs = [(a, b) for a, b in list(zip(pressures, pressures[1:]))]

out_dir = Path('/Users/jarl/2d-discharge-nn/data/interpolation')
feather_outdir = Path('/Users/jarl/2d-discharge-nn/data/interpolation_feather')

# iterate over intermediate voltages
for pressure in tqdm(pressures, desc=' pressure', position=0):
    for start_voltage, end_voltage in v_pairs:
        start_df = get_data_table(data_folder, start_voltage, pressure)
        end_df = get_data_table(data_folder, end_voltage, pressure)
        parameters = list(start_df.columns[4:])
        inter_voltage = (start_voltage + end_voltage)/2

        # ensure X and Y are the same across both files
        assert start_df.iloc[:,2:4].all(axis=None) == end_df.iloc[:,2:4].all(axis=None)

        parameter_arrays = [interpolation(parameter, 'voltage') for parameter in parameters]

        inter_df = pd.concat(parameter_arrays, axis=1)
        inter_df = pd.concat([inter_df, start_df.iloc[:,2:4]], axis=1)
        inter_df['Vpp [V]'] = inter_voltage
        inter_df['P [Pa]'] = pressure

        assert (start_df.columns == end_df.columns).all()
        # inter_df[list(start_df.columns)].to_csv(out_dir / f'{voltage:.0f}V{pressure:.0f}Pa.csv', index=False) # csv
        inter_df[list(start_df.columns)].to_feather(feather_outdir / f'{inter_voltage:.0f}Vpp{pressure:03.0f}Pa_node.feather')

# iterate over intermediate pressures
for voltage in tqdm(voltages, desc='voltage'):
    for start_pressure, end_pressure in p_pairs:
        start_df = get_data_table(data_folder, voltage, start_pressure)
        end_df = get_data_table(data_folder, voltage, end_pressure)
        parameters = list(start_df.columns[4:])
        inter_pressure = (start_pressure + end_pressure)/2

        assert start_df.iloc[:,2:4].all(axis=None) == end_df.iloc[:,2:4].all(axis=None)

        parameter_arrays = [interpolation(parameter, 'pressure') for parameter in parameters]

        inter_df = pd.concat(parameter_arrays, axis=1)
        inter_df = pd.concat([inter_df, start_df.iloc[:,2:4]], axis=1)
        inter_df['Vpp [V]'] = voltage
        inter_df['P [Pa]'] = inter_pressure

        assert (start_df.columns == end_df.columns).all()
        # inter_df[list(start_df.columns)].to_csv(out_dir / f'{voltage:.0f}V{pressure:.0f}Pa.csv', index=False)
        inter_df[list(start_df.columns)].to_feather(feather_outdir / f'{voltage:.0f}Vpp{inter_pressure:03.0f}Pa_node.feather')
