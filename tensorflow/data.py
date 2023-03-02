#coding: utf-8

import posixpath
import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import Normalize
from scipy import interpolate


def read_all_data(fldr_path, voltages, pressures):
    file_count = 0
    for voltage in voltages:
        for pressure in pressures:
            file_name = '{0:d}Vpp_{1:03d}Pa_node.dat'.format(voltage,pressure)
            file_path = posixpath.join(fldr_path, file_name)
            if os.path.exists(file_path):
                data = read_file(file_path)
                file_count += 1
            else:
                continue
            
            data = attach_VP_columns(data, voltage, pressure)
            data_table = data if file_count==1 else pd.concat([data_table,data], ignore_index=True)
    return data_table


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


def data_preproc(data_line): # ! obsolete !
    ret_data_line = []
    for n,one_val in enumerate(data_line, start=1):
        if n in (3,6,7,8,9): # "potential (V)", "Ne (#/m^-3)", "Ar+ (#/m^-3)", "Nm (#/m^-3)", "Te (eV)"
            if one_val!=0:
                one_val = np.log10(one_val)
            else:
                ret_data_line = False
                break
        ret_data_line.append(one_val)
    return ret_data_line


def attach_VP_columns(data, voltage, pressure):
    num_data_points = len(data)
    vp_columns = [[voltage, pressure] for n in range(num_data_points)]
    vp_columns = pd.DataFrame(vp_columns, columns=['Vpp [V]', 'P [Pa]'])
    return vp_columns.join(data)


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


if __name__ == '__main__':
    import sys
    
    # path to data directory
    data_fldr_path = './data/avg_data'
    
    # mode selection by number of arguments of this script file
    argc = len(sys.argv)
    if argc==1:
        # mode 1: draw 2D graphs and save them for all following combinations of Vpp and P
        #voltages  = [200, 300, 400, 500] # V
        #pressures = [  5,  10,  30,  45, 60, 80, 100, 120] # Pa
        voltages = [300]
        pressures = [60]
        fldr_path_to_save = './data/graphs'
    elif argc==3:
        # mode 2: draw 2D graphs for one specific combination of Vpp and P, then display them on your screen
        #         Vpp and P must be given as arguments of this script file like below.
        #         # python data.py Vpp P
        voltages  = [eval(sys.argv[1])] # V
        pressures = [eval(sys.argv[2])] # Pa
    else:
        print('error: invalid num of arguments.')
        raise Exception
    
    avg_data = read_all_data(data_fldr_path, voltages, pressures)
    if len(avg_data)==0:
        print('error: no data available.')
        raise Exception
    #print(avg_data)
    
    num_obj_vars = 7
    
    param_col_labels = avg_data.columns[-num_obj_vars:]
    t, f = True, False
    if argc==1:
        if t:
            for voltage in voltages:
                for pressure in pressures:
                    avg_data_VP = avg_data[(avg_data['Vpp [V]']==voltage) & (avg_data['P [Pa]']==pressure)]
                    if len(avg_data_VP)==0:
                        continue
                    print('Vpp = {0:d} [V], P = {1:3d} [Pa]'.format(voltage,pressure))
                    for n,param_col_label in enumerate(param_col_labels, start=1):
                        fig_file = posixpath.join(fldr_path_to_save, 'fig_{0:d}V_{1:d}Pa_{2:d}.png'.format(voltage,pressure,n))
                        draw_a_2D_graph(avg_data_VP, param_col_label, file_path=fig_file)
        if f:
            # print the value range
            for param_col_label in param_col_labels:
                max_val = np.max(avg_data[param_col_label].values)
                min_val = np.min(avg_data[param_col_label].values)
                print(param_col_label,':', min_val, '-', max_val)
    elif argc==3:
        fldr_path_to_save = '.'
        if t:
            for n,param_col_label in enumerate(param_col_labels, start=1):
                #fig_file = posixpath.join(fldr_path_to_save, 'fig_{0:d}_org.png'.format(n))
                fig_file = None
                draw_a_2D_graph(avg_data, param_col_label, file_path=fig_file)
        if f:
            interp_data = get_interp_data(avg_data, param_col_labels, save_fig=f)
        

