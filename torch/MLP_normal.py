"""
Explore different data normalization for the MLP.

23 May 17:00
author @jarl
"""

import re
from datetime import datetime as dt
from tqdm import tqdm
from pathlib import Path

import numpy as np
import pandas as pd
import dask.dataframe as dd

def create_dataframe(file: Path, V_list: list, P_list: list) -> pd.DataFrame:
    """Create DataFrame from simulation data.

    Args:
        file (Path): Path to the .dat files containing simulation data.
        V_list (list): list of voltages
        P_list (list): list of pressures

    Returns:
        pd.DataFrame: DataFrame for one pair of (V, P)
    """
    name = file.stem
    title = name.split('_')  # split the file at underscores, this should return [VVpp, PPa, node]
    # since the voltages and pressures follow xxxVpp, yyyPa:
    V = float(title[0][:3])
    P = float(title[1][:3])

    V_list.append(V)
    P_list.append(P)

    with open(file, 'r') as f:
        data = []
        for n, line in enumerate(f, 1):
            if n==1:
                line = line.strip()
                line = re.findall(r'"[^"]*"', line)
                column_labels = [var_name.replace('"', '') for var_name in line]
            elif n==2:
                continue  # skip the second line
            else:
                data_line = [eval(data) for data in line.split()]
                if len(data_line)==4:
                    break
                data.append(data_line)
    df = pd.DataFrame(data, columns=column_labels).drop(columns=['Ex (V/m)', 'Ey (V/m)'])
    df['Vpp [V]'] = V  # add V column
    df['P [Pa]'] = P  # add P column
    
    return df


def save_metadata(out_dir: Path):
    """Save readable metadata.

    Args:
        out_dir (Path): Path to where the data is saved.
    """
    with open(out_dir / 'metadata.txt', 'w') as f:
            f.write(f'Name: {name}\n')
            f.write(f'Created: {now}\n')
            f.write(f'Normalization: none\n')
            f.write(f'Voltages [V]: {set(V_list)}\n')
            f.write(f'Pressures [Pa]: {set(P_list)}\n')
            f.write('\n*** end of file ***\n')


if __name__ == '__main__':
    # import data
    root = Path('/Users/jarl/2d-discharge-nn/data/')
    data_dir = root/'avg_data'
    out_dir = root/'dataframes'
    data_list = data_dir.rglob('*.dat')

    name = 'simulation_data'
    V_list = []
    P_list = []

    now = dt.now().strftime('%d-%m-%Y %H:%M')
    df_list = [create_dataframe(file, V_list, P_list) for file in tqdm(data_list)]
    df = pd.concat(df_list, ignore_index=True)

    # just to check
    df_list[0].head()

    df.to_feather(out_dir/(name + '.feather'))
    save_metadata(out_dir)

    # normalize data
