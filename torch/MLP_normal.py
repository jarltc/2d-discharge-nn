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


class MeshDataset:
    def __init__(self, normalization, minmax):
        self.voltages = []
        self.pressures = []
        self.normalization = normalization
        self.test_v = 300
        self.test_p = 60
        self.minmax = minmax
        self.name = None
        self.savedate = None

        # TODO: train/test/validation splitting
        # normalization scheme (default/tobi)
        # scale data
        # read data
        # create feather file if it doesn't exist
        # save metadata
        # move to data_helpers

        def create_dataframe(self, file: Path) -> pd.DataFrame:
            """Create DataFrame from simulation data.

            Args:
                file (Path): Path to the .dat files containing simulation data.
                V_list (list): list of voltages
                P_list (list): list of pressures

            Returns:
                pd.DataFrame: DataFrame for one pair of (V, P)
            """
            # split the file at underscores, this should return [XVpp, YPa, node]
            # since the voltages and pressures follow xxxVpp, yyyPa, we can use list indexing
            title = file.stem.split('_')
            V = float(title[0][:3])
            P = float(title[1][:3])

            self.voltages.append(V)
            self.pressures.append(P)

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

        def save_feather(self, name, out_dir):
            df.to_feather(out_dir/(name + '.ftr'))
            save_metadata(out_dir)

        def save_metadata(self, out_dir: Path):
            """Save readable metadata.

            Args:
                out_dir (Path): Path to where the data is saved.
            """
            with open(out_dir / 'metadata.txt', 'w') as f:
                    f.write(f'Name: {self.name}\n')
                    f.write(f'Created: {self.savedate}\n')
                    f.write(f'Normalization: {self.normalization}\n')
                    f.write(f'Voltages [V]: {set(self.voltages)}\n')
                    f.write(f'Pressures [Pa]: {set(self.pressures)}\n')
                    f.write(f'Excluded set (V, P): {self.test_v}, {self.test_p}')
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

    # TODO: exclude 300, 60
    now = dt.now().strftime('%d-%m-%Y %H:%M')
    df_list = [create_dataframe(file, V_list, P_list) for file in tqdm(data_list)]
    df = pd.concat(df_list, ignore_index=True)

    # normalize data
