# Plotting functions module

import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib import gridspec
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'Arial'

import pandas as pd
import numpy as np
import xarray as xr

import torch
import torch.nn as nn
import torchvision

import time
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler

from data_helpers import mse, get_data

def triangulate(df: pd.DataFrame):   
    """
    Create triangulation of the mesh grid, which is passed to tricontourf.
    
    Uses Delaunay triangulation.

    Parameters
    ----------
    df : DataFrame
        DataFrame with X and Y values for the triangulation.

    Returns
    -------
    triangles : matplotlib.tri.triangulation.Triangulation
        Triangulated grid.

    """
    x = df['x'].to_numpy()*100
    y = df['y'].to_numpy()*100
    triangles = matplotlib.tri.Triangulation(x, y)
    
    return triangles


# from data.py
def get_cbar_range(param_col_label):
    if param_col_label=='potential (V)':
        cmin = 0.0
        cmax = 210
    elif param_col_label=='Ex (V/m)':
        cmin = -150000
        cmax =   75000
    elif param_col_label=='Ey (V/m)':
        cmin = -150000
        cmax =  150000
    elif param_col_label=='Ne (#/m^-3)':
        cmin = 0.0
        cmax = 1.6e16
    elif param_col_label=='Ar+ (#/m^-3)':
        cmin = 7.5e10
        cmax = 1.6e16
    elif param_col_label=='Nm (#/m^-3)':
        cmin = 3.3e11
        cmax = 8.9e18
    elif param_col_label=='Te (eV)':
        cmin =  0.0
        cmax = 12.0
    return cmin, cmax


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


def draw_a_2D_graph(avg_data, param_col_label, file_path=None, set_cbar_range=True, c_only=False, lin=False):
    # data
    
    units = {'potential (V)'    :' ($\mathrm{10 V}$)', 
             'Ne (#/m^-3)'      :' ($10^{14}$ $\mathrm{m^{-3}}$)',
             'Ar+ (#/m^-3)'     :' ($10^{14}$ $\mathrm{m^{-3}}$)', 
             'Nm (#/m^-3)'      :' ($10^{16}$ $\mathrm{m^{-3}}$)',
             'Te (eV)'          :' (eV)'}

    
    matplotlib.rcParams['font.family'] = 'Arial'
    x = avg_data.X.values.reshape(-1,1)*100
    y = avg_data.Y.values.reshape(-1,1)*100
    z = avg_data[param_col_label].values.reshape(-1,1)
    
    cmap=plt.cm.viridis
    
    # settings for drawing
    # plt.rcParams['font.family'] = 'serif'
    fig = plt.figure(figsize=(3.3,7), dpi=200)
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    if set_cbar_range:
        #cmin, cmax = get_cbar_range(param_col_label)
        cmin, cmax = get_cbar_range_300V_60Pa(param_col_label, lin=lin)
        sc = ax.scatter(x, y, c=z, cmap=cmap, alpha=0.5, s=2, linewidths=0, vmin=cmin, vmax=cmax)
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
    fig.tight_layout()
    
    if file_path==None:
        plt.show()
    else:
        fig.savefig(file_path)
    plt.close('all')


def save_history_graph(history: list, out_dir: Path):
    matplotlib.rcParams['font.family'] = 'Arial'
    x  = np.array(history)

    plt.rcParams['font.size'] = 12 # 12 is the default size
    plt.rcParams['xtick.minor.size'] = 2
    plt.rcParams['ytick.minor.size'] = 2
    fig, ax = plt.subplots(figsize=(6.0,6.0), dpi=200)
    
    # axis labels
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    
    ax.plot(x, color='green', lw=1.0, label='train_error')
    
    # set both x_min and y_min as zero
    _, x_max = ax.get_xlim()
    _, y_max = ax.get_ylim()
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    
    ax.xaxis.get_ticklocs(minor=True)
    ax.yaxis.get_ticklocs(minor=True)
    ax.minorticks_on()
    
    plt.legend()
    ax.grid()
    plt.tight_layout()
    
    # save the figure
    # if param=='mae':
    #     file_name = 'history_graph_mae.png'
    # elif param=='loss':
    file_name = 'history_graph_loss.png'
    file_path = out_dir / file_name
    fig.savefig(file_path)


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


def quickplot(df:pd.DataFrame, data_dir=None, triangles=None, nodes=None, mesh=False):
    """Quick plot of all 5 parameters.

    This makes a plot for just the predictions, but it might help to have
    the actual simulation results for comparison (TODO).
    Args:
        df (pd.DataFrame): DataFrame of only the predictions (no features).
        data_dir (Path, optional): Path to where the model is saved. Defaults to None.
        grid (bool, optional): _description_. Defaults to False.
        triangles (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    matplotlib.rcParams['font.family'] = 'Arial'
    cmap = plt.cm.viridis

    # check whether to plot a filled contour or mesh points
    
    fig = plt.figure(dpi=200, figsize=(9, 4), constrained_layout=True)
    grid = ImageGrid(
        fig, 111, nrows_ncols=(1, len(df.columns)), 
        axes_pad=0.5, label_mode="L", share_all=True,
        cbar_location="right", cbar_mode="each", cbar_size="7%", cbar_pad="5%")

    titles = [column.split()[0] for column in df.columns]

    if mesh:
        for i, column in enumerate(df.columns):
            ax = grid[i]
            cax = grid.cbar_axes[i]
            ax.set_aspect('equal')
            cmin, cmax = get_cbar_range_300V_60Pa(column, lin=True)
            sc = ax.scatter(nodes['x'], nodes['y'], c=df[column], 
                                    cmap=cmap,
                                    norm=colors.Normalize(vmin=cmin, vmax=cmax),
                                    s=0.2)
            cax.colorbar(sc)
            draw_apparatus(ax)
            ax.set_xlim(0,20)
            ax.set_ylim(0,70.9)
            ax.set_title(titles[i])

    else:
        for i, column in enumerate(df.columns):
            ax = grid[i]
            cax = grid.cbar_axes[i]
            ax.set_aspect('equal')
            cmin, cmax = get_cbar_range_300V_60Pa(column, lin=True)
            tri = ax.tricontourf(triangles, df[column], levels=36, 
                                    cmap=cmap, vmin=cmin, vmax=cmax)
            cax.colorbar(tri)
            draw_apparatus(ax)
            ax.set_title(titles[i])
        

    if data_dir is not None:
        if mesh:
            fig.savefig(data_dir/'quickplot_mesh.png', bbox_inches='tight')
        else:
            fig.savefig(data_dir/'quickplot.png', bbox_inches='tight')

    return fig


def correlation(prediction: pd.DataFrame, targets: pd.DataFrame, scores=None, scores_list=None, out_dir=None):
    """Plot correlation between true values and predictions.

    Args:
        prediction (pd.DataFrame): DataFrame of predicted values.
        targets (pd.DataFrame): DataFrame of true values.
        scores (pd.DataFrame): Scores containing the r2.
        out_dir (Path, optional): Path to save file. Defaults to None.
    """
    assert list(prediction.columns) == list(targets.columns)

    prediction = prediction.copy()
    targets = targets.copy()

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
        if scores is None:
            r2 = round(scores_list[i], 2)
        else: 
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

    diff = 100 * ((py - ty) / np.abs(ty)) 
    titles = [column.split()[0] for column in diff.columns]

    tX['x'] = tX['x']*100
    tX['y'] = tX['y']*100

    # ranges = {'potential (V)': (-5, 5), 
    #           'Ne (#/m^-3)' : (-8, 8),
    #           'Ar+ (#/m^-3)' : (-9, 9),
    #           'Nm (#/m^-3)' : (-20, 50),
    #           'Te (eV)' : (-50, 50)}
    
    cmap = plt.get_cmap('coolwarm')

    # fig, ax = plt.subplots(ncols=5, dpi=200, figsize=(12, 4), constrained_layout=True)
    fig = plt.figure(dpi=200, figsize=(8, 4), constrained_layout=True)
    gs = gridspec.GridSpec(1, 6, width_ratios=[1, 1, 1, 1, 1, 0.1], figure=fig)
    
    for i, column in enumerate(ty.columns):
        ax = fig.add_subplot(gs[0, i])
        sc = ax.scatter(tX['x'], tX['y'], c=diff[column], cmap=cmap, 
                        #    norm=colors.Normalize(vmin=ranges[column][0], vmax=ranges[column][1]), 
                           norm=colors.Normalize(vmin=-100, vmax=100),
                           s=0.2) 
        draw_apparatus(ax)
        ax.set_title(titles[i])
        ax.set_aspect('equal')
        ax.set_xlim(0,20)
        ax.set_ylim(0,70.9)
        draw_apparatus(ax)
    
    cax = fig.add_subplot(gs[0, 5])
    cbar = plt.colorbar(sc, extend='both', cax=cax)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(r'% difference', size=8)

    fig.savefig(out_dir/'difference.png', bbox_inches='tight')

    return fig


def plot_comparison_ae(reference: np.ndarray, prediction: torch.tensor, model:nn.Module, 
                       out_dir=None, is_square=False, mode='reconstructing', resolution=32): 
    """Create plot comparing the reference data with its autoencoder reconstruction.

    Args:
        reference (np.ndarray): Reference dataset.
        prediction (torch.tensor): Tensor reshaped to match the encoding shape.
        model (nn.Module): Autoencoder model whose decoder used to make predictions.
        out_dir (Path, optional): Output directory. Defaults to None.
        is_square (bool, optional): Switch for square image and full rectangle.
            Defaults to False.
        mode (str, optional): Switch for 'reconstructing' or 'predicting' (plots title). 
            Defaults to 'reconstructing'.

    Returns:
        float: Evaluation time.
        scores: List of reconstruction MSE
    """
    if is_square:
        figsize = (6, 3)
        extent = [0, 20, 35, 55]
    else:
        figsize = (10, 5)
        extent =[0, 20, 0, 70.7]

    fig = plt.figure(dpi=300, layout='constrained')
    
    grid = ImageGrid(fig, 111,  # similar to fig.add_subplot(142).
                     nrows_ncols=(2, 5), axes_pad=0.0, label_mode="L", share_all=True,
                     cbar_location="right", cbar_mode="single", cbar_size="5%", cbar_pad='5%')

    cbar_ranges = [(reference[0, i, :, :].min(),
                    reference[0, i, :, :].max()) 
                    for i in range(5)]

    with torch.no_grad():
        start = time.time_ns()
        reconstruction = torchvision.transforms.functional.crop(
            model.decoder(prediction), 0, 0, resolution, resolution).cpu().numpy() 
        end = time.time_ns()

    eval_time = round((end-start)/1e-6, 2)

    # plot the figures
    for i, ax in enumerate(grid):
        if i <= 4:
            j = i
            org = ax.imshow(reference[0, i, :, :], origin='lower', extent=extent, aspect='equal',
                            vmin=cbar_ranges[j][0], vmax=cbar_ranges[j][1], cmap='magma')
            draw_apparatus(ax)
        else:
            j = i-5
            rec = ax.imshow(reconstruction[0, i-5, :, :], origin='lower', extent=extent, aspect='equal',
                            vmin=cbar_ranges[j][0], vmax=cbar_ranges[j][1], cmap='magma')
            draw_apparatus(ax)
        grid.cbar_axes[0].colorbar(org)

    # set font sizes and tick stuff
    for ax in grid:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(3))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))

        ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(2))
        ax.tick_params(axis='both', labelsize=8)

    # record scores
    scores = []
    for i in range(5):
        score = mse(reference[0, i, :, :], reconstruction[0, i, :, :])
        scores.append(score)

    if out_dir is not None:
        fig.savefig(out_dir/f'test_comparison1.png', bbox_inches='tight')

    return eval_time, scores

def ae_correlation(reference, prediction, out_dir):
    from sklearn.metrics import r2_score
    scores = []
    columns = ['pot', 'ne', 'ni', 'nm', 'te']
    prediction_cols = []
    reference_cols = []

    prediction = prediction.cpu().numpy()
    
    for i, column in enumerate(columns):
        ref_series = pd.Series(reference[0, i, :, :].flatten(), name=column)
        pred_series = pd.Series(prediction[0, i, :, :].flatten(), name=column)
        scores.append(r2_score(reference[0, i, :, :].flatten(), 
                               prediction[0, i, :, :].flatten()))
        reference_cols.append(ref_series)
        prediction_cols.append(pred_series)
    
    ref_df = pd.DataFrame({k: v for k, v in zip(columns, reference_cols)})
    pred_df = pd.DataFrame({k: v for k, v in zip(columns, prediction_cols)})

    correlation(pred_df, ref_df, scores_list=scores, out_dir=out_dir)

    return scores

def slices(model, kind='mesh', out_dir=None):
    columns = ['potential (V)', 'Ne (#/m^-3)', 'Ar+ (#/m^-3)', 'Nm (#/m^-3)', 'Te (eV)']
    if kind == 'mesh':
        # test v and p (scaled)
        v = 300.0
        p = 60.0
        testV = (v - 200.0)/(500.0 - 200.0)
        testP = (p - 5.0)/(120.0 - 5.0)

        v = [200., 300., 400., 500.]
        p = [  5.,  10.,  30.,  45.,  60.,  80., 100., 120.]

        # scale distances (assumes max ranges are 0.2 and .707)
        # x_scaler = (x - xmin)/(xmax - xmin), x/xmax if xmin=0
        x_scaler = 1/0.2
        y_scaler = 1/0.707
        
        # horizontal line
        xhs = np.linspace(0, 0.2, 1000)
        yh = 0.44  # m
        horizontal = np.array([np.array([testV, testP, xh*x_scaler, yh*y_scaler]) for xh in xhs.tolist()])

        # vertical line
        xv = 0.115  # m
        yvs = np.linspace(0, 0.707, 1000)
        vertical = np.array([np.array([testV, testP, xv*x_scaler, yv*y_scaler]) for yv in yvs.tolist()])

        # create tensor of x, y points for both horizontal and vertical slices
        horizontal = torch.FloatTensor(horizontal)
        vertical = torch.FloatTensor(vertical)

        # predict x, y points from model and plot
        model.eval()
        with torch.no_grad():
            vertical_prediction = pd.DataFrame(model(horizontal).numpy(), columns=columns)
            horizontal_prediction = pd.DataFrame(model(vertical).numpy(), columns=columns)

    # elif kind == 'image':
    #     # using model (mlp+decoder), predict images
    #     # get slice of image (how??) xr.dataarray?
    #     image.plot()
    else: print('how you manage to break this cuh 💀')

    colors = ['#d20f39', '#df8e1d', '#40a02b', '#04a5e5', '#8839ef']
    fig, ax = plt.subplots(dpi=300)
    
    # horizontal plot
    for i, column in enumerate(columns):
        ax.plot(xhs*100, horizontal_prediction[column], color=colors[i], label=column)
        ax.grid()
        ax.legend()
        ax.set_ylabel('Scaled magnitude')
        ax.set_xlabel('x [cm]')

    # add simulation data to the plots (use image data)
    if out_dir is not None:
        fig.savefig(out_dir/'slices.png', bbox_inches='tight')
