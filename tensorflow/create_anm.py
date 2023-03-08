#coding:utf-8

import os, sys
import datetime
import pickle
import posixpath
import shutil
import math
import warnings

import glob
from PIL import Image

import numpy as np
import pandas as pd
from scipy import interpolate

import sklearn
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.animation as ani
import matplotlib.patches as pat

import data, do_regr


warnings.simplefilter('ignore')


def get_cbar_range(fixed_param_str, fixed_param_val, data_dir):
    if fixed_param_str=='v':
        min_file_name = f'{int(fixed_param_val):d}Vpp_005Pa_node.dat'
        max_file_name = f'{int(fixed_param_val):d}Vpp_120Pa_node.dat'
        min_cond = {'v':fixed_param_val, 'p':5}
        max_cond = {'v':fixed_param_val, 'p':120}
    elif fixed_param_str=='p':
        min_file_name = f'200Vpp_{int(fixed_param_val):03d}Pa_node.dat'
        max_file_name = f'500Vpp_{int(fixed_param_val):03d}Pa_node.dat'
        min_cond = {'v':200, 'p':fixed_param_val}
        max_cond = {'v':500, 'p':fixed_param_val}
    else:
        print('error: invalid param_str:', fixed_param_str)
        raise Exception
    
    min_file_path = posixpath.join(data_dir, min_file_name)
    max_file_path = posixpath.join(data_dir, max_file_name)
    
    if os.path.exists(min_file_path) and os.path.exists(max_file_path):
        min_data = data.read_all_data(data_dir, [min_cond['v']], [min_cond['p']])
        max_data = data.read_all_data(data_dir, [max_cond['v']], [max_cond['p']])
        
        cbar_range = {}
        for (min_col_name,min_col_vals),(max_col_name,max_col_vals) in zip(min_data.iteritems(),max_data.iteritems()):
            cbar_range.update({min_col_name:{'min':min_col_vals.values.min()*1.15,
                                             'max':max_col_vals.values.max()*0.85}})
    else:
        cbar_range = {
            'potential (V)': {'min':0.0    ,'max':210   },
            'Ex (V/m)'     : {'min':-150000,'max':75000 },
            'Ey (V/m)'     : {'min':-150000,'max':150000},
            'Ne (#/m^-3)'  : {'min':0.0    ,'max':1.6e16},
            'Ar+ (#/m^-3)' : {'min':7.5e10 ,'max':1.6e16},
            'Nm (#/m^-3)'  : {'min':3.3e11 ,'max':8.9e18},
            'Te (eV)'      : {'min':0.0    ,'max':12.0  }
        }
    
    return cbar_range


def get_XY_and_col_labels(data_dir):
    # get coordinate data
    vlt = 300 # V
    prs = 60  # Pa
    
    avg_data = do_regr.create_dummy_data_table(data_dir, vlt, prs)
    avg_data = avg_data.drop(columns=['Ex (V/m)','Ey (V/m)'])
    
    return avg_data[['X','Y']], avg_data.iloc[:,4:].columns


def create_dirs(anm_dir, num_params):
    for n in range(1, num_params+1):
        param_dir = anm_dir + f'/param_{n:02d}'
        if not os.path.exists(param_dir):
            os.mkdir(param_dir)


def set_a_graph(param_col_label):
    fig = plt.figure(figsize=(5.5,7.0))
    ax = fig.add_subplot(111)
    
    plt.title(param_col_label)
    
    # x axis
    plt.xlabel('r [cm]')
    plt.xlim( 0, 21) # 0-21cm
    plt.xticks([0,5,10,15,20])
    
    # y axis
    plt.ylabel('z [cm]')
    plt.ylim(20, 55) # 0-72cm
    
    return fig, ax


def show_text_in_graph(ax, vlt, prs):
    textbox_dic = {
        'facecolor':'white',
        'edgecolor':'black',
        'linewidth':1}
    ax.text(
        0.05, 0.93, # text position
        f'{vlt:5.1f} Vpp, {prs:5.1f} Pa',
        fontsize=18, bbox=textbox_dic,
        transform=ax.transAxes)


def draw_apparatus(ax):
    def edge_unit_conv(edges):
        u = 1e-1 # unit conv. mm -> cm
        return [(xy[0]*u,xy[1]*u) for xy in edges]
    
    pt_colors = {'fc':'gray', 'ec':'black'}
    
    edges = [(0,453), (0,489), (95,489), (95,487), (40,487), (40,453)]
    patch_top = pat.Polygon(xy=edge_unit_conv(edges), fc=pt_colors['fc'], ec=pt_colors['ec'])
    ax.add_patch(patch_top)
    
    edges = [(0,0), (0,415), (95,415), (95,395), (90,395), (90,310), (120,310), (120,277), (90,277), (90,0)]
    patch_bottom = pat.Polygon(xy=edge_unit_conv(edges), fc=pt_colors['fc'], ec=pt_colors['ec'])
    ax.add_patch(patch_bottom)
    
    edges = [(122,224), (122,234), (185,234), (185,224)]
    patch_float = pat.Polygon(xy=edge_unit_conv(edges), fc=pt_colors['fc'], ec=pt_colors['ec'])
    ax.add_patch(patch_float)


def plot_and_save_figs(pred_rslts, anm_dir, fig_n, cbar_range):
    # coordinate
    x = pred_rslts.X.values.reshape(-1,1)*100
    y = pred_rslts.Y.values.reshape(-1,1)*100
    
    for param_n,param_col_label in enumerate(pred_rslts.columns[4:], start=1):
        # settings for drawing
        plt.rcParams['font.family'] = 'serif'
        fig, ax = set_a_graph(param_col_label)
        
        # plot
        sc = ax.scatter(
            x, y, c=pred_rslts[param_col_label].values,
            cmap=plt.cm.jet, alpha=0.5,
            s=10, linewidths=0,
            vmin=cbar_range[param_col_label]['min'],
            vmax=cbar_range[param_col_label]['max'])
        
        # color bar
        cbar = plt.colorbar(sc)
        cbar.minorticks_off()
        
        # text in the graph [*** Vpp, *** Pa]
        show_text_in_graph(ax, pred_rslts['Vpp [V]'][0], pred_rslts['P [Pa]'][0])
        
        # draw the apparatus
        #draw_apparatus(ax)
        
        plt.tight_layout()
        
        # save
        param_fig_file = posixpath.join(anm_dir+f'/param_{param_n:02d}', f'{fig_n:03d}.png')
        fig.savefig(param_fig_file)
        
        plt.close('all')


def interp_plot_and_save(pred_rslts, anm_dir, fig_n, cbar_range):
    # create a new mesh
    N = 250
    grid_x, grid_y = np.meshgrid(
        np.linspace(0, pred_rslts.X.values.max()*100, N+1),
        np.linspace(0, pred_rslts.Y.values.max()*100, N+1))
    
    for param_n,param_col_label in enumerate(pred_rslts.columns[4:], start=1):
        # settings for drawing
        plt.rcParams['font.family'] = 'serif'
        fig, ax = set_a_graph(param_col_label)
        
        # interpolating
        interpolated_data = interpolate.griddata(
            pred_rslts[['X','Y']]*100,
            pred_rslts[param_col_label].values,
            (grid_x, grid_y),
            method='cubic').reshape(N+1,N+1)
        
        # plot
        sc = ax.pcolormesh(
            grid_x, grid_y, interpolated_data,
            cmap=plt.cm.jet,
            norm=Normalize(
                vmin=cbar_range[param_col_label]['min'],
                vmax=cbar_range[param_col_label]['max']),
            shading='gouraud')
        
        # color bar
        cbar = plt.colorbar(sc)
        cbar.minorticks_off()
        
        # text in the graph [*** Vpp, *** Pa]
        show_text_in_graph(ax, pred_rslts['Vpp [V]'][0], pred_rslts['P [Pa]'][0])
        
        # draw the apparatus
        draw_apparatus(ax)
        
        plt.tight_layout()
        
        # save
        param_fig_file = posixpath.join(anm_dir+f'/param_{param_n:02d}', f'{fig_n:03d}.png')
        fig.savefig(param_fig_file)
        
        plt.close('all')


def show_progress(num_figs, fig_n):
    prog_pcnt = fig_n/(num_figs-1)*100
    if prog_pcnt in np.linspace(0,100,6):
        d = datetime.datetime.today()
        print(f'  {int(prog_pcnt):3d}%...', d.strftime('%H:%M:%S'), flush=True)


def create_anm_gif_file(fig_list, anm_dir, n):
    fig = plt.figure(figsize=(5.5,7.0)) # <-- should be same as the graph size
    
    plt.axis('off')
    
    graph_figs = [[plt.imshow(Image.open(one_fig))] for one_fig in fig_list]
    
    plt.tight_layout()
    
    animation = ani.ArtistAnimation(fig, graph_figs, interval=300, repeat_delay=1000)
    gif_ani_file = posixpath.join(anm_dir, f'param_{n:02d}.gif')
    #animation.save(gif_ani_file, writer='Pillow')
    animation.save(gif_ani_file, writer='imagemagick')
    
    plt.clf()


################################################################


if __name__ == '__main__':
    d = datetime.datetime.today()
    print('started on', d.strftime('%Y-%m-%d %H:%M:%S'), '\n')
    
    print('versions')
    print('python    :', sys.version)
    print('tensorflow:', tf.__version__)
    print('keras     :', keras.__version__)
    print('sklearn   :', sklearn.__version__)
    print()
    
    # -------------------------------------------------------
    
    data_dir = './data/avg_data'
    
    model_dir = './created_models/2021-10-29_2120'
    
    #voltage = 300 # V
    pressure = 100 # Pa
    
    create_fig_files = False
    create_anm_gif   = True
    
    # -------------------------------------------------------
    
    if create_fig_files:
        # voltages and pressures
        if 'voltage' in locals():
            P_min, P_max = 5, 120 # Pa
            step = 0.5
            pressures = np.linspace(P_min, P_max, int((P_max-P_min)/step)+1)
            num_figs = pressures.size
            voltages  = [voltage]
            anm_dir = model_dir + f'/regr_anm_{voltage:d}Vpp'
            print(f'param: voltage={voltage:.1f}Vpp, pressure={P_min:.1f}-{P_max:.1f}Pa')
            cbar_range = get_cbar_range('v', voltage, data_dir)
        elif 'pressure' in locals():
            V_min, V_max = 200, 500 # V
            step = 1.5
            voltages  = np.linspace(V_min, V_max, int((V_max-V_min)/step)+1)
            num_figs = voltages.size
            pressures = [pressure]
            anm_dir = model_dir + f'/regr_anm_{pressure:d}Pa'
            print(f'param: voltage={V_min:.1f}-{V_max:.1f}Vpp, pressure={pressure:.1f}Pa')
            cbar_range = get_cbar_range('p', pressure, data_dir)
        print(f'num figs: {num_figs:d}\n')
        
        # create a directory
        if not os.path.exists(anm_dir):
            os.mkdir(anm_dir)
        
        # file back up
        shutil.copyfile(posixpath.join('.',sys.argv[0]), posixpath.join(anm_dir,sys.argv[0]))
        
        # collect some info
        XY, param_col_labels = get_XY_and_col_labels(data_dir)
        
        # load NN model
        model = keras.models.load_model(model_dir+'/model')
        print()
        
        # create figure files
        first_loop = True
        fig_count = 1
        for vlt in voltages:
            for prs in pressures:
                # create descriptor table
                X = data.attach_VP_columns(XY, vlt, prs)
                tX = do_regr.create_descriptors_for_regr(X, model_dir)
                stX = do_regr.scale_for_regr(tX, model_dir)
                
                # get predicted values
                spy = model.predict(stX)
                py = do_regr.inv_scale(spy, param_col_labels, model_dir)
                py = pd.DataFrame(py, columns=param_col_labels)
                py = do_regr.data_postproc(py)
                
                pred_rslts = pd.concat([X,py], axis='columns')
                
                # create directories
                if first_loop:
                    print('\nprogress')
                    create_dirs(anm_dir, len(param_col_labels))
                    first_loop = False
                
                # draw figures and save them
                #plot_and_save_figs(pred_rslts, anm_dir, fig_count, cbar_range)
                interp_plot_and_save(pred_rslts, anm_dir, fig_count, cbar_range)
                
                show_progress(num_figs, fig_count)
                fig_count += 1
    
    # create animation gif files (easy, but low quality...)
    if create_anm_gif:
        # file back up
        if not create_fig_files:
            if 'voltage' in locals():
                anm_dir = model_dir + f'/regr_anm_{voltage:d}Vpp'
            elif 'pressure' in locals():
                anm_dir = model_dir + f'/regr_anm_{pressure:d}Pa'
            file_name, ext = sys.argv[0].split('.')[-2:]
            new_file_name = file_name + '_gif.' + ext
            shutil.copyfile(posixpath.join('.',sys.argv[0]), posixpath.join(anm_dir,new_file_name))
        
        if 'param_col_labels' not in locals():
            _, param_col_labels = get_XY_and_col_labels(data_dir)
        
        print('creating gif anime files...')
        for n,param_col_label in enumerate(param_col_labels, start=1):
            fig_dir = anm_dir + f'/param_{n:02d}'
            fig_list = glob.glob(fig_dir + '\*.png')
            
            if len(fig_list)==0:
                print('error: no graph file available.')
                break
            
            print(f'param {n:02d}: {param_col_label:s}')
            create_anm_gif_file(fig_list, anm_dir, n)
    
    d = datetime.datetime.today()
    print('\nfinished on', d.strftime('%Y-%m-%d %H:%M:%S'))