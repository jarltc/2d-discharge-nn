#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 11:59:41 2022

@author: jarl
"""

import matplotlib.pyplot as plt
from matplotlib import colors as cm
import matplotlib.cm as cm
import pandas as pd
from pathlib import Path

folder = Path('/home/jarl/Desktop/Research/2D_NN/data/300V_60Pa_interpolation')
file = folder/'data_table.csv'
out_dir = Path('/home/jarl/Desktop/Research/2D_NN/pred_plots')

df = pd.read_csv(file)

output_cols =  [5,6,7,8]

for col in output_cols:
    fig, ax = plt.subplots(figsize=(8,4), dpi=200)
    cmap = cm.magma
    heat = ax.scatter(df.X, df.Y, c=df.iloc[:,col], alpha=0.5,
                      linewidths=0, s=0.5, cmap=cmap)
    ax.set_aspect('equal')
    ax.set_title(df.columns[col], fontsize=9)
    ax.set_ylabel('z [cm]')
    ax.set_xlabel('r [cm]')
    plt.colorbar(heat, extend='max', pad=0.015)
    file_name = (df.columns[col].partition('(')[0] + '.png')

    # fig.savefig(out_dir/file_name, bbox_inches='tight')