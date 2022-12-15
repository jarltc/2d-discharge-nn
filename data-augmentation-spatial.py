#coding: utf-8
"""
Data augmentation.

Create datasets of linear interpolation from mesh grid to linear grid.

* edit get_interp_data() and update the old one in data.py
@author: jarl
Created on Thu 15 Dec 2022
"""

import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# to delete
from matplotlib.colors import Normalize
from scipy import interpolate
import posixpath

def get_interp_data(avg_data, param_col_labels, plot_now=True, save_fig=False):
    # create new mesh
    N = 300
    x_max = avg_data.X.values.max()
    y_max = avg_data.Y.values.max()
    grid_x, grid_y = np.meshgrid(np.linspace(0,x_max,N), np.linspace(0,y_max,N))
    
    grid = []
    for xi,yi in zip(grid_x,grid_y):
        for xij,yij in zip(xi,yi):
            grid.append([xij,yij])
    data_table = pd.DataFrame(grid, columns=['X','Y'])
    
    if plot_now:
        plt.rcParams['font.family'] = 'serif'
    
    for param_count,param_col_label in enumerate(param_col_labels, start=1):
        # interpolating
        z = avg_data[param_col_label].values.reshape(-1,1)
        zmin, zmax = z.min(), z.max()
        print('param', param_count, '...', end='', flush=True)
        new_z_3D = interpolate.griddata(avg_data[['X','Y']], z, (grid_x,grid_y), method='cubic')
        new_z_2D = new_z_3D.reshape(N,N)
        new_z_vals = new_z_2D.reshape(-1,1)
        #new_z_vals = np.nan_to_num(new_z_vals, nan=0)
        print('created.', flush=True)
        new_z = pd.DataFrame(new_z_vals, columns=[param_col_label])
        data_table = pd.concat([data_table,new_z], axis=1)
        
        # drawing graphs
        if plot_now:
            fig = plt.figure(figsize=(5.5,10.0))
            ax = fig.add_subplot(111)
            sc = ax.pcolormesh(
                grid_x, grid_y, new_z_2D,
                cmap=plt.cm.jet, norm=Normalize(vmin=zmin,vmax=zmax),
                shading='gouraud')
            cbar = plt.colorbar(sc)
            plt.title(param_col_label)
            plt.xlim(0.0, 0.21)
            plt.ylim(0.0, 0.72)
            plt.tight_layout()
            if save_fig:
                file_path = posixpath.join(fldr_path_to_save, 'fig_{0:d}.png'.format(param_count))
                fig.savefig(file_path)
            else:
                plt.show()
            plt.close('all')
    #data_table.to_csv('./data_int.csv', index=False)
    return data_table


###### main #######
root = Path(os.getcwd())
data_folder = root/'data'/'avg_data'

voltages = [200, 300, 400, 500]
pressures = [5, 10, 30, 45, 60, 80, 100, 120]

out_dir = root/'data'/'interpolation_datasets'
if not os.path.exists(out_dir): os.mkdir(out_dir)

file = data_folder/'300Vpp_060Pa_node.dat'
# for file in data_folder.rglob(*):
