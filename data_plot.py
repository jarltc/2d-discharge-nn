# -*- coding: utf-8 -*-
"""
Create continuous plot for simulation results (reference).
(original just had colored grid points)

* to do: convert to module for plotting purposes

@author: jarl
"""

import os
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import re
import pandas as pd
# import data
import numpy as np

matplotlib.rcParams['font.family'] = 'Arial'
file_path = '/Users/jarl/2d-discharge-nn/data/avg_data/300Vpp_060Pa_node.dat'

def read_file(file_path):
    """
    Import .dat file into a DataFrame.

    Parameters
    ----------
    file_path : string
        Path to the file.

    Returns
    -------
    DataFrame
        Dataframe containing the data.

    """
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


def draw_a_2D_graph(avg_data, param_col_label, triangles, file_path=None,   
                    set_cbar_range=True, c_only=False, lin=False):          
    """
    Plot the 2D graph.

    Parameters
    ----------
    avg_data : DataFrame
        DataFrame containing the data.
    param_col_label : str
        Column label to be passed to avg_data.
    triangles : matplotlib.tri.triangulation.Triangulation
        Triangulation of the mesh from triangles().
    file_path : str, optional
        Path to the output directory. Plots to the console if None.
    set_cbar_range : bool, optional
        Specify colorbar ranges. The default is True.
    c_only : bool, optional
        no idea. The default is False.
    lin : bool, optional
        Set to True to plot linearly scaled data. The default is False.

    Returns
    -------
    None.

    """
    # change units on fig title if lin scale
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
    fig = plt.figure(figsize=(3.3, 7), dpi=200)
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    if set_cbar_range:
        #cmin, cmax = get_cbar_range(param_col_label)
        cmin, cmax = get_cbar_range_300V_60Pa(param_col_label, lin=lin) 
        sc = plt.tricontourf(triangles, avg_data[param_col_label], levels=36,
                             cmap=cmap, vmin=cmin, vmax=cmax)
    else:
        # sc = ax.scatter(x, y, c=z, cmap=cmap, alpha=0.5, s=2, linewidths=0)
        sc = plt.tricontourf(triangles, avg_data[param_col_label],
                             levels=36, cmap=cmap)
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


def get_cbar_range_300V_60Pa(param_col_label, lin=False):
    """
    Set range values (max, min) for the colorbar.

    Parameters
    ----------
    param_col_label : str
        Column label (target parameters).
    lin : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    cmin : float
        Minimum value for the colorbar.
    cmax : float
        Maximum value for the colorbar.

    """
    if lin:
        column_ranges = {'potential (V)':   (0.0, 9.8),
                         'Ex (V/m)'     :   (-80000,  19000),  # unused
                         'Ey (V/m)'     :   (-68000, 72000),  # unused
                         'Ne (#/m^-3)'  :   (0.0, 58),
                         'Ar+ (#/m^-3)' :   (4.5e-3, 58),
                         'Nm (#/m^-3)'  :   (3.2e-3, 88),
                         'Te (eV)'      :   (0.0, 6.0)}
        cmin, cmax = column_ranges[param_col_label]
                         
    else:
        column_ranges = {'potential (V)':   (0.0, 98.0),
                         'Ex (V/m)'     :   (-80000,  19000),  # unused
                         'Ey (V/m)'     :   (-68000, 72000),  # unused
                         'Ne (#/m^-3)'  :   (0.0, 5.8e15),
                         'Ar+ (#/m^-3)' :   (4.5e11, 5.8e15),
                         'Nm (#/m^-3)'  :   (3.2e13, 8.8e17),
                         'Te (eV)'      :   (0.0, 6.0)}
        cmin, cmax = column_ranges[param_col_label]
        
    return cmin, cmax


def triangulate(df):   
    """
    Create triangulation of the mesh grid*, which is passed to tricontourf.
    
    Uses Delaunay triangulation.
    * need to check if this is working right

    Parameters
    ----------
    df : DataFrame
        DataFrame with X and Y values for the triangulation.

    Returns
    -------
    triangles : matplotlib.tri.triangulation.Triangulation
        Triangulated grid.

    """
    df['X'].update(df['X']*100)
    df['Y'].update(df['Y']*100)
    triangles = matplotlib.tri.Triangulation(df.X, df.Y)
    
    return triangles


def electrodes(ax):
    """
    Plot the electrodes. Uses rough estimations of positions using pixel information.
    
    Need to get the actual measurements from Sarah.
    
    Parameters
    ----------
    ax : Axes
        Axes where the electrodes are to be added.

    Returns
    -------
    None
    
    """
    from matplotlib.patches import Rectangle
    
    def px_to_cm(px):
        onecm = 16.7
        return px/onecm
    
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
    
    # process data (if linear), hide if not
    for column in df.iloc[:,2:].columns:
        exp = round(np.log10(df[column].values.mean()), 0) - 1.0
        df[column].update(df[column]/(10**exp))
    
    out_dir = Path('/Users/jarl/2d-discharge-nn/data/reference')  
    
    df['X'].update(df['X']*100)
    df['Y'].update(df['Y']*100)
    triangles = matplotlib.tri.Triangulation(df.X, df.Y)
    # mask1 = matplotlib.tri.TriAnalyzer(triangles).get_flat_tri_mask(0.05)
    # triangles.set_mask(mask1)
    
    for n,p_param in enumerate(df.columns[2:], start=1): # figs
        param_name = p_param.split(' ')[0]
        # draw_a_2D_graph(df, p_param, triangles, lin=True, 
        #                 file_path=out_dir/f'{param_name}-filled2.png')
        draw_a_2D_graph(df, p_param, triangles, lin=True)
    
    # fig2, ax2 = plt.subplots(dpi=400)
    # ax2.set_aspect('equal')
    # ax2.triplot(triangles, lw=0.2)
    
    # scipy griddata also works but is much slower than triangle contours
    # y = np.linspace(df.Y.min(), df.Y.max(), 10000)
    # x = np.linspace(df.X.min(), df.X.max(), 10000)
    # xv, yv = np.meshgrid(x, y)
    # points = df.iloc[:,:2].values
    # grid = griddata(points, df['potential (V)'], (xv, yv), method='linear')
    # grid = np.flipud(grid)
    # ax.imshow(grid)
    


