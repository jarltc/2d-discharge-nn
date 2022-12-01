# -*- coding: utf-8 -*-
"""
Spyder Editor

Create continuous plot for simulation results (reference)
"""

import os
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import re
import pandas as pd
import data
import numpy as np

matplotlib.rcParams['font.family'] = 'Arial'
file_path = '/Users/jarl/2d-discharge-nn/data/avg_data/300Vpp_060Pa_node.dat'

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


def draw_a_2D_graph(avg_data, param_col_label, triangles, file_path=None,   #  TODO
                    set_cbar_range=True, c_only=False, lin=False):          #  TODO
    # data
    units = {'potential (V)'    :' ($\mathrm{10 V}$)', 
             'Ne (#/m^-3)'      :' ($10^{14}$ $\mathrm{m^{-3}}$)',
             'Ar+ (#/m^-3)'     :' ($10^{14}$ $\mathrm{m^{-3}}$)', 
             'Nm (#/m^-3)'      :' ($10^{16}$ $\mathrm{m^{-3}}$)',
             'Te (eV)'          :' (eV)'}
    
    x = avg_data.X.values.reshape(-1,1)*100
    y = avg_data.Y.values.reshape(-1,1)*100
    z = avg_data[param_col_label].values.reshape(-1,1)
    
    cmap=plt.cm.viridis
    
    # settings for drawing
    # plt.rcParams['font.family'] = 'serif'
    fig = plt.figure(figsize=(3.3, 7), dpi=200)
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    if set_cbar_range:
        #cmin, cmax = get_cbar_range(param_col_label)
        cmin, cmax = get_cbar_range_300V_60Pa(param_col_label, lin=lin)
        # sc = ax.scatter(x, y, c=z, cmap=cmap, alpha=0.5, s=2, linewidths=0, vmin=cmin, vmax=cmax)
        sc = plt.tricontourf(triangles, avg_data[param_col_label], levels=36, cmap=cmap, vmin=cmin, vmax=cmax)
    else:
        sc = ax.scatter(x, y, c=z, cmap=cmap, alpha=0.5, s=2, linewidths=0)
    cbar = plt.colorbar(sc)
    cbar.minorticks_off()
    
    if lin:
        title = param_col_label.split(' ')[0] + units[param_col_label]
    else:
        title = param_col_label
    
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('r [cm]')
    ax.set_ylabel('z [cm]')
    ax.set_xlim(0, 21)
    ax.set_ylim(0, 72)
    
    electrodes(ax)
    
    fig.tight_layout()
    
    if file_path==None:
        plt.show()
    else:
        fig.savefig(file_path)
    plt.close('all')


def get_cbar_range_300V_60Pa(param_col_label, lin=False): ## TODO: fix this
    if lin:
        if param_col_label=='potential (V)':
            cmin =  0.0
            cmax = 9.8
        elif param_col_label=='Ex (V/m)':
            cmin = -80000
            cmax =  19000
        elif param_col_label=='Ey (V/m)':
            cmin = -68000
            cmax =  72000
        elif param_col_label=='Ne (#/m^-3)':
            cmin = 0.0
            cmax = 58
        elif param_col_label=='Ar+ (#/m^-3)':
            cmin = 4.5e-3
            cmax = 58
        elif param_col_label=='Nm (#/m^-3)':
            cmin = 3.2e-3
            cmax = 88
        elif param_col_label=='Te (eV)':
            cmin = 0.0
            cmax = 6.0
    else:
        if param_col_label=='potential (V)':
            cmin =  0.0
            cmax = 98.0
        elif param_col_label=='Ex (V/m)':
            cmin = -80000
            cmax =  19000
        elif param_col_label=='Ey (V/m)':
            cmin = -68000
            cmax =  72000
        elif param_col_label=='Ne (#/m^-3)':
            cmin = 0.0
            cmax = 5.8e15
        elif param_col_label=='Ar+ (#/m^-3)':
            cmin = 4.5e11
            cmax = 5.8e15
        elif param_col_label=='Nm (#/m^-3)':
            cmin = 3.2e13
            cmax = 8.8e17
        elif param_col_label=='Te (eV)':
            cmin = 0.0
            cmax = 6.0
    return cmin, cmax

def triangulate(df):
    df['X'].update(df['X']*100)
    df['Y'].update(df['Y']*100)
    triangles = matplotlib.tri.Triangulation(df.X, df.Y)
    
    return triangles

def electrodes(ax):
    from matplotlib.patches import Rectangle
    
    def px_to_cm(px):
        onecm = 16.7
        return px/onecm

    # anchors = [(0, 48.18), (0, 44.82), (0, 0), (0, 40.82), (0, 27.33), (12.18, 22.12)]
    # ws = [0.12, 3.35, 41.29, 2.18, 3.47, 1.23]    
    # hs = [9.23, 3.82, 8.76, 9.23, 11.76, 7.06]
    
    anchors = [(0, 813), (0, 756), (0, 0), (0, 661), (0, 463), (209, 380)]
    anchors = [(px_to_cm(x), px_to_cm(y)) for x, y in anchors]
    
    hs = [2, 57, 691, 30, 59, 15]
    hs = [px_to_cm(x) for x in hs]
    
    ws = [157, 65, 149, 157, 200, 98]  
    ws = [px_to_cm(x) for x in ws]
    
    rectangles = [Rectangle(anchors[i], ws[i], hs[i], 
                            facecolor='white', edgecolor=None, 
                            zorder=2) for i in range(len(anchors))]
    
    for rectangle in rectangles:
        ax.add_patch(rectangle)


if __name__ == '__main__':
    df = read_file(file_path).drop(columns=['Ex (V/m)', 'Ey (V/m)'])
    for column in df.iloc[:,2:].columns:
        exp = round(np.log10(df[column].values.mean()), 0) - 1.0
        df[column].update(df[column]/(10**exp))
    
    
    out_dir = Path('/Users/jarl/2d-discharge-nn/data/reference')
    
    units = {'potential (V)'    :' ($\mathrm{10 V}$)', 
             'Ne (#/m^-3)'      :' ($10^{14}$ $\mathrm{m^{-3}}$)',
             'Ar+ (#/m^-3)'     :' ($10^{14}$ $\mathrm{m^{-3}}$)', 
             'Nm (#/m^-3)'      :' ($10^{16}$ $\mathrm{m^{-3}}$)',
             'Te (eV)'          :' (eV)'}
    
    
    df['X'].update(df['X']*100)
    df['Y'].update(df['Y']*100)
    triangles = matplotlib.tri.Triangulation(df.X, df.Y)
    
    for n,p_param in enumerate(df.columns[2:], start=1): # figs
        param_name = p_param.split(' ')[0]
        # draw_a_2D_graph(df, p_param, triangles, lin=True, 
        #                 file_path=out_dir/f'{param_name}-filled2.png')
        draw_a_2D_graph(df, p_param, triangles, lin=True)

