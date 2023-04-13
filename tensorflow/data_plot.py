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
import matplotlib.patches as pat
from matplotlib import colors
from sklearn.preprocessing import MinMaxScaler
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


def draw_a_2D_graph(avg_data, param_col_label, triangles, file_path=None, set_cbar_range=True,   
                    on_grid=False, lin=False, X_mesh=None, Y_mesh=None):          
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

    def data_plot_core(avg_data, param_col_label):

        if on_grid:
            image_array = avg_data[param_col_label].to_numpy().reshape(X_mesh.shape)
            if set_cbar_range:  #
                cmin, cmax = get_cbar_range_300V_60Pa(param_col_label, lin=lin) 
                sc = ax.imshow(image_array, cmap=cmap, vmin=cmin, vmax=cmax, 
                               aspect='equal', extent=[0, X_mesh.max()*100, 0, Y_mesh.max()*100], 
                               origin='lower')
            else:
                sc = ax.imshow(image_array, cmap=cmap, aspect='equal', 
                               extent=[0, X_mesh.max()*100, 0, Y_mesh.max()*100], origin='lower')
        else:
            if set_cbar_range:
                cmin, cmax = get_cbar_range_300V_60Pa(param_col_label, lin=lin)
                sc = plt.tricontourf(triangles, avg_data[param_col_label], levels=36, 
                                 cmap=cmap, vmin=cmin, vmax=cmax)
            else:
                sc = plt.tricontourf(triangles, avg_data[param_col_label], levels=36, cmap=cmap)

        return sc


    cmap = plt.cm.viridis

    # change units on fig title if lin scale
    units = {'potential (V)'    :' ($\mathrm{10 V}$)', 
             'Ne (#/m^-3)'      :' ($10^{14}$ $\mathrm{m^{-3}}$)',
             'Ar+ (#/m^-3)'     :' ($10^{14}$ $\mathrm{m^{-3}}$)', 
             'Nm (#/m^-3)'      :' ($10^{16}$ $\mathrm{m^{-3}}$)',
             'Te (eV)'          :' (eV)'}
    
    if not on_grid:  # if predicting on mesh
        # x = avg_data.X.values.reshape(-1,1)*100
        # y = avg_data.Y.values.reshape(-1,1)*100
        # z = avg_data[param_col_label].values.reshape(-1,1)
        data = avg_data.set_index(['X', 'Y'])

    else: data = avg_data
    
    # settings for drawing
    fig = plt.figure(figsize=(3.3, 7), dpi=200)
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')

    sc = data_plot_core(data, param_col_label)

    cbar = plt.colorbar(sc)
    cbar.minorticks_off()
    
    if lin:
        title = param_col_label.split(' ')[0] + units[param_col_label]
    else:
        title = param_col_label
    
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('r [cm]')
    ax.set_ylabel('z [cm]')
    ax.set_xlim(0, X_mesh.max()*100)
    ax.set_ylim(0, Y_mesh.max()*100)
    
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


def triangulate(df: pd.DataFrame):   
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


def draw_apparatus(ax):
    def edge_unit_conv(edges):
        u = 1e-1 # unit conv. mm -> cm
        return [(xy[0]*u,xy[1]*u) for xy in edges]
    
    pt_colors = {'fc':'white', 'ec':'black'}
    
    edges = [(0,453), (0,489), (95,489), (95,487), (40,487), (40,453)]
    patch_top = pat.Polygon(xy=edge_unit_conv(edges), fc=pt_colors['fc'], ec=pt_colors['ec'])
    ax.add_patch(patch_top)
    
    edges = [(0,0), (0,415), (95,415), (95,395), (90,395), (90,310), (120,310), (120,277), (90,277), (90,0)]
    patch_bottom = pat.Polygon(xy=edge_unit_conv(edges), fc=pt_colors['fc'], ec=pt_colors['ec'])
    ax.add_patch(patch_bottom)
    
    edges = [(122,224), (122,234), (185,234), (185,224)]
    patch_float = pat.Polygon(xy=edge_unit_conv(edges), fc=pt_colors['fc'], ec=pt_colors['ec'])
    ax.add_patch(patch_float)


def difference_plot(tX: pd.DataFrame, py: pd.DataFrame, ty: pd.DataFrame, out_dir: Path):
    """Plot the difference between predictions and true values. (py - ty)

    Args:
        tX (pd.DataFrame): DataFrame of (V, P, X, Y)
        py (pd.DataFrame): DataFrame of predictions.
        ty (pd.DataFrame): DataFrame of corresponding true values.
        out_dir (Path): Directory to save the figure.
    """
    assert list(py.columns) == list(ty.columns)

    # TODO: move elsewhere
    units = {'potential (V)'    :' ($\mathrm{10 V}$)', 
            'Ne (#/m^-3)'      :' ($10^{14}$ $\mathrm{m^{-3}}$)',
            'Ar+ (#/m^-3)'     :' ($10^{14}$ $\mathrm{m^{-3}}$)', 
            'Nm (#/m^-3)'      :' ($10^{16}$ $\mathrm{m^{-3}}$)',
            'Te (eV)'          :' (eV)'}

    diff = py - ty
    titles = [column.split()[0] for column in diff.columns]

    tX['X'] = tX['X']*100
    tX['Y'] = tX['Y']*100

    fig, ax = plt.subplots(ncols=5, dpi=200, figsize=(12, 4))
    fig.subplots_adjust(wspace=0.2)
    
    for i, column in enumerate(ty.columns):
        sc = ax[i].scatter(tX['X'], tX['Y'], c=diff[column], cmap='coolwarm', 
                           norm=colors.CenteredNorm(), s=1)
        plt.colorbar(sc)
        ax[i].set_title(titles[i] + ' ' + units[column])
        ax[i].set_aspect('equal')
        ax[i].set_xlim(0,20)
        ax[i].set_ylim(0,70.9)

    fig.savefig(out_dir/'difference.png', bbox_inches='tight')

    return fig


def all_plot(tX: pd.DataFrame, py: pd.DataFrame, ty: pd.DataFrame, out_dir: Path, simulation=False):
    """Plot the predictions as five subplots.

    Args:
        tX (pd.DataFrame): DataFrame of (V, P, X, Y) [m] -> [cm]
        py (pd.DataFrame): DataFrame of predictions.
        ty (pd.DataFrame): DataFrame of corresponding true values.
        out_dir (Path): Directory to save the figure.
    """
    assert list(py.columns) == list(ty.columns)
    if simulation:
        py = ty

    # TODO: move elsewhere
    units = {'potential (V)'    :' ($\mathrm{10 V}$)', 
            'Ne (#/m^-3)'      :' ($10^{14}$ $\mathrm{m^{-3}}$)',
            'Ar+ (#/m^-3)'     :' ($10^{14}$ $\mathrm{m^{-3}}$)', 
            'Nm (#/m^-3)'      :' ($10^{16}$ $\mathrm{m^{-3}}$)',
            'Te (eV)'          :' (eV)'}

    triangles = triangulate(tX[['X', 'Y']])
    titles = [column.split()[0] for column in ty.columns]

    fig, ax = plt.subplots(ncols=5, dpi=200, figsize=(12, 4))
    fig.subplots_adjust(wspace=0.2)
    
    for i, column in enumerate(ty.columns):
        cmin, cmax = get_cbar_range_300V_60Pa(column, lin=True)
        sc = ax[i].tricontourf(triangles, py[column], levels=36, 
                            cmap='viridis', vmin=cmin, vmax=cmax)
        plt.colorbar(sc)
        draw_apparatus(ax[i])
        ax[i].set_title(titles[i] + ' ' + units[column])
        ax[i].set_aspect('equal')
        ax[i].set_xlim(0,20)
        ax[i].set_ylim(0,70.9)

    if simulation:
        fig.savefig(out_dir/'ref_quickplot.png', bbox_inches='tight')
    else:
        fig.savefig(out_dir/'quickplot.png', bbox_inches='tight')

    return fig


def correlation(prediction: pd.DataFrame, targets: pd.DataFrame, scores: pd.DataFrame, out_dir=None):
    """Plot correlation between true values and predictions.

    Stolen from torch's plot.py.

    Args:
        prediction (pd.DataFrame): DataFrame of predicted values.
        targets (pd.DataFrame): DataFrame of true values.
        scores (pd.DataFrame): Scores containing the r2.
        out_dir (Path, optional): Path to save file. Defaults to None.
    """
    assert list(prediction.columns) == list(targets.columns)

    prediction = prediction.copy()
    targets = targets.copy()

    # catppuccin latte palette
    colors = ['#d20f39', '#df8e1d', '#40a02b', '#04a5e5', '#8839ef']

    fig, ax = plt.subplots(dpi=200)
    
    # customize axes
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.set_ylabel('Predicted')
    ax.set_xlabel('True')

    # plot 1:1 line
    x = np.linspace(0, 1, 1000)
    ax.plot(x, x, ls='--', c='k')

    for i, column in enumerate(prediction.columns):
        # transform with minmax to normalize between (0, 1)
        scaler = MinMaxScaler()
        scaler.fit(targets[column].values.reshape(-1, 1))
        scaled_targets = scaler.transform(targets[column].values.reshape(-1, 1))
        scaled_predictions = scaler.transform(prediction[column].values.reshape(-1, 1))

        # get correlation score
        r2 = round(scores[column].iloc[3], 2)

        # set label
        label = f'{column.split()[0]}: {r2}'

        ax.scatter(scaled_targets, scaled_predictions, s=1, marker='.',
                   color=colors[i], alpha=0.15, label=label)

    legend = ax.legend(markerscale=4, fontsize='small')
    for lh in legend.legendHandles: 
        lh.set_alpha(1)
    
    if out_dir is not None:
        fig.savefig(out_dir/'correlation.png', bbox_inches='tight')


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
    


