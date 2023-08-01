#coding:utf-8
"""
Model test regression code.

Perform regression of model on test (V, P). New code written to accommodate
linearly scaled data.

* to do: move redundant functions to a utils module

"""


import os, sys
import datetime
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras

import data
import data_plot
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-m', '--mesh', action='store_false', help='Interpolate on mesh.')
args = vars(parser.parse_args())

def get_data_table(data_dir, voltage, pressure):
    file_name = '{0:d}Vpp_{1:03d}Pa_node.dat'.format(voltage,pressure)
    file_path = data_dir / file_name
    if os.path.exists(file_path):
        avg_data = data.read_all_data(data_dir, [voltage], [pressure])
    else:
        avg_data = create_dummy_data_table(data_dir, voltage, pressure)
    return avg_data


def create_dummy_data_table(data_dir, voltage, pressure):
    ext_data = data.read_all_data(data_dir, [300], [60])
    ext_data = ext_data.drop(columns=['Vpp [V]','P [Pa]'])
    
    XY_df = ext_data.iloc[:,:2]
    ext_data = ext_data.drop(columns=['X','Y'])
    
    nan_mat = np.zeros_like(ext_data.values)
    nan_mat[:,:] = np.nan
    Nan_df = pd.DataFrame(nan_mat, columns= ext_data.columns)
    
    dummy_df = pd.concat([XY_df,Nan_df], axis=1)
    
    dummy_df = data.attach_VP_columns(dummy_df, voltage, pressure)
    
    return dummy_df


def create_descriptors_for_regr(data_table, model_dir):
    def get_var(model_dir, var_file_name):
        file_path = model_dir / var_file_name
        if os.path.exists(file_path):
            with open(file_path, 'rb') as pf:
                return pickle.load(pf)
        else:
            print('file missing:', var_file_name)
            return None
    
    dsc_linlog = get_var(model_dir, 'dsc_linlog.pkl')
    if dsc_linlog is None:
        dsc_linlog = ['lin', 'lin', 'lin', 'lin']
    
    dsc_pows = get_var(model_dir, 'dsc_pows.pkl')
    if dsc_pows is None:
        dsc_pows = {'V':1, 'P':1, 'x':1, 'y':1}
    
    pow_labels = ('V', 'P', 'x', 'y')
    linlog_n = 0
    dsc_col_labels = []
    for col_n,(col_name,col_vals) in enumerate(data_table.iteritems()):
        pow_label = pow_labels[col_n]
        for pow_n in range(1,dsc_pows[pow_label]+1):
            tmp_col = col_vals.values.reshape(-1,1)**pow_n
            
            if dsc_linlog[linlog_n]=='log':
                tmp_col = np.log10(tmp_col)
            linlog_n += 1
            
            dsc_table = tmp_col if col_n==0 and pow_n==1 else np.hstack([dsc_table,tmp_col])
            
            label = pow_label if pow_n==1 else pow_label+'**{0:d}'.format(pow_n)
            dsc_col_labels.append(label)
    
    return pd.DataFrame(dsc_table, columns=dsc_col_labels)


def get_scale_exp():
    if os.path.exists(model_dir / 'train_metadata.pkl'):
        scale_exp = metadata['parameter_exponents']
    elif os.path.exists(model_dir / 'scale_exp.pkl'):
        def get_scale(path):
            with open(path, 'rb') as sf:
                scale_exp = pickle.load(sf) 
            return scale_exp

        scale_exp_file = model_dir / 'scale_exp.pkl'
        scale_exp = get_scale(scale_exp_file)
    else:
        print('Scaler unavailable, assuming data is logarithmically scaled.')
        scale_exp = None
    
    return scale_exp


def scale_for_regr(data_table, model_dir):
    for n,column in enumerate(data_table.columns, start=1):
        one_data_col = data_table[column].values.reshape(-1,1)
        scaler_file = model_dir / 'scalers' / 'xscaler_{0:02d}.pkl'.format(n)
        with open(scaler_file, 'rb') as sf:
            xscaler = pickle.load(sf)
            xscaler.clip = False
            scaled_data = xscaler.transform(one_data_col)
        scaled_data_table = scaled_data if n==1 else np.hstack([scaled_data_table,scaled_data])
    
    scaled_data_table = pd.DataFrame(scaled_data_table, columns=data_table.columns)
    
    return scaled_data_table


def inv_scale(scaled_data, columns, model_dir):
    num_columns = len(columns)
    for n in range(num_columns):
        scaler_file = model_dir / 'scalers' / 'yscaler_{0:02d}.pkl'.format(n+1)
        with open(scaler_file, 'rb') as sf:
            yscaler = pickle.load(sf)
            yscaler.clip = False
            inv_scaled_data = yscaler.inverse_transform(scaled_data[:,n].reshape(-1,1))
        inv_scaled_data_table = inv_scaled_data if n==0 else np.hstack([inv_scaled_data_table,inv_scaled_data])
    
    inv_scaled_data_table = pd.DataFrame(inv_scaled_data_table, columns=columns)
    
    return inv_scaled_data_table


def data_postproc(data_table, lin=False):
    """
    Reverse log-scaling if model is trained on log-data.
    Can also reverse scaling for lin data (if needed) using lin.

    Parameters
    ----------
    data_table : DataFrame
        DataFrame of predicted values (py).
    
    lin : bool
        Switch if data is not linearly scaled. Defaults to False.

    Returns
    -------
    DataFrame
        DataFrame of unscaled prediction values.

    """
    scale_exp = get_scale_exp()

    trgt_params = ('potential (V)', 'Ne (#/m^-3)', 'Ar+ (#/m^-3)', 'Nm (#/m^-3)', 'Te (eV)')
    
    for col_n,(col_name,col_vals) in enumerate(data_table.iteritems(), start=1):
        vals = col_vals.values.reshape(-1,1)
        
        # unscale log-data if not lin
        if lin:
            tmp_col = vals * (10**scale_exp[col_n-1]) if col_name in trgt_params else vals
        else:
            tmp_col = 10**vals if col_name in trgt_params else vals
        
        post_proced_table = tmp_col if col_n==1 else np.hstack([post_proced_table,tmp_col])
        
    return pd.DataFrame(post_proced_table, columns=data_table.columns)


def save_pred_vals(tX, py, rslt_dir=None):
    if rslt_dir == None:
        raise Exception('save_pred_vals: No result directory specified!')
    pred_rslts = pd.concat([tX,py], axis='columns')
    rslt_file = rslt_dir / 'predicted_rslts.csv'
    pred_rslts.to_csv(rslt_file, index=False)


def print_scores(ty, py, regr_dir=None):
    global spy

    # if on_grid:
    #     stX = scale_for_regr(tX, model_dir)
    #     spy = model.predict(stX)

    if minmax_y:
        py = inv_scale(spy, ty.columns, model_dir)
        py = pd.DataFrame(py, columns=ty.columns)
    else:
        py = pd.DataFrame(spy, columns=ty.columns)

    if not lin:  # if data is logarithmic, add postprocessing step
        py = data_postproc(py)

    def print_scores_core(exp):
        e_params = ('Ne (#/m^-3)', 'Ar+ (#/m^-3)', 'Nm (#/m^-3)')
        scores = []
        for col_n,col_label in enumerate(ty.columns, start=1):
            print('**** {0:d}: {1:s} ****'.format(col_n,col_label), file=exp)
            
            ty_col = ty[col_label].values.flatten()
            py_col = py[col_label].values.flatten()
            
            mae   = mean_absolute_error(ty_col, py_col)
            rmse  = np.sqrt(mean_squared_error(ty_col,py_col))
            r2    = r2_score(ty_col, py_col)
            ratio = rmse/mae

            scores.append(np.array([[mae], [rmse], [ratio], [r2]]))
            
            if col_label in e_params: ## TODO
                print('MAE      = {0:.6f}'.format(mae), file=exp)
                print('RMSE     = {0:.6f}'.format(rmse), file=exp)
            else:
                print('MAE      = {0:.6f}'.format(mae), file=exp)
                print('RMSE     = {0:.6f}'.format(rmse), file=exp)
            print('R2 score = {0:.6f}'.format(r2), file=exp)
            print('RMSE/MAE = {0:.6f}'.format(ratio), file=exp)
            print(file=exp)

        return scores
    
    scores = print_scores_core(sys.stdout)
    
    if regr_dir is not None:
        file_path = regr_dir / 'scores.txt'
        with open(file_path, 'w') as f:
            print_scores_core(f)

    scores = np.hstack(scores)
    scores_df = pd.DataFrame(scores, columns=list(ty.columns))

    return scores_df


def ty_proc(ty):
    '''
    Process target data (simulation .dat file).
    
    Target data is in original scaling, whereas the training predictions are
    scaled down by 10^n. Exponents used to scale the data down for 
    training (n) are stored as a pickle file in the model folder. 
    The pickle file is a list of powers, which are used to reverse the scaling.

    Parameters
    ----------
    ty : DataFrame
        Target data.

    Returns
    -------
    ty: DataFrame
        Returns scaled target data (ty.columns[i]* 10^(-n[i]) ).

    '''
    scale_exp = get_scale_exp()
    
    # trgt_params = ['potential (V)', 'Ne (#/m^-3)', 'Ar+ (#/m^-3)', 'Nm (#/m^-3)', 'Te (eV)']
    for i in range(len(scale_exp)):
        ty.update(ty.iloc[:, i]/(10**scale_exp[i]))
    
    return ty


################################################################
if __name__ == '__main__':
    # -------------------------------------------------------
    root = Path(os.getcwd())  # root folder where everything is saved
    on_grid = False #args['mesh']  # raised flag stores false: defaults to True (pixels)
    d = datetime.datetime.today()
    print('started on', d.strftime('%Y-%m-%d %H:%M:%S'), '\n')
    
    print('versions')
    print('python    :', sys.version)
    print('tensorflow:', tf.__version__)
    print('keras     :', keras.__version__)
    print('sklearn   :', sklearn.__version__)
    # print()
    
    # -------------------------------------------------------
    
    # perform regression with these parameters
    voltage  = 300 # V
    pressure =  60 # Pa
    
    data_dir = root / 'data' / 'avg_data'  # simulation data
    model_dir = Path(input('Model directory: '))
    model = keras.models.load_model(model_dir / 'model')

    step = 0.001  # controls grid spacing for grid data

    # infer info from model metadata
    if os.path.exists(model_dir / 'train_metadata.pkl'):
        with open(model_dir / 'train_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        minmax_y = metadata['is_target_scaled']
        lin = metadata['scaling']
        name = metadata['name']
    else:
        print('Metadata unavailable: using defaults lin=True, minmax_y=True\n')
        lin = True
        minmax_y = True
        name = model_dir.name
 
    print('\nLoaded model ' + name)
    # -------------------------------------------------------
    
    # get simulation data
    avg_data = get_data_table(data_dir, voltage, pressure)
    avg_data = avg_data.drop(columns=['Ex (V/m)','Ey (V/m)'])
    
    # split data into target x and y; tX is created from mesh grid points from simulation data
    # tX = create_descriptors_for_regr(avg_data.iloc[:,:4], model_dir)
    tX = avg_data.iloc[:, :4].copy()
    tX['x**2'] = tX['X']**2
    tX['y**2'] = tX['Y']**2

    tX.rename(columns={'Vpp [V]'  : 'V',
                        'P [Pa]'  : 'P',
                        'X'       : 'x', 
                        'Y'       : 'y'}, inplace=True)

    tX = tX[['V', 'P', 'x', 'x**2', 'y', 'y**2']]
    
    # create a grid of datapoints with 1x1 mm resolution, x2, y2, v, and p
    y = np.arange(0, 0.707 + step, step)
    x = np.arange(0, 0.2 + step, step)
    X, Y = np.meshgrid(x, y)

    tX_grid = pd.DataFrame({'x':X.flatten(), 'y':Y.flatten()})
    tX_grid['V'] = voltage
    tX_grid['P'] = pressure
    tX_grid['x**2'] = tX_grid['x']**2
    tX_grid['y**2'] = tX_grid['y']**2
    tX_grid = tX_grid[['V', 'P', 'x', 'x**2', 'y', 'y**2']]

    # scale target data if necessary (if model predicts linearly)
    ty = ty_proc(avg_data.iloc[:,4:]) if lin else avg_data.iloc[:,4:]
    
    # scale tX for use in model.predict()
    # if on_grid: stX = scale_for_regr(tX_grid, model_dir)
    # else: stX = scale_for_regr(tX, model_dir)
    stX = scale_for_regr(tX, model_dir)
    
    # display conds
    print('model_dir:', model_dir)
    print('obj vars : ', end='')
    for n,column in enumerate(ty.columns, start=1):
        print(column, end='')
        if n<len(ty.columns):
            print(', ', end='')
        else:
            print()
    print('regr cond: {0:d} V, {1:d} Pa'.format(voltage,pressure))
    print()
    
    # predict from target x, scale if necessary
    spy = model.predict(stX)
    if minmax_y:
        py = inv_scale(spy, ty.columns, model_dir)
        py = pd.DataFrame(py, columns=ty.columns)
    else:
        py = py = pd.DataFrame(spy, columns=ty.columns)

    if not lin:  # if data is logarithmic, add postprocessing step
        py = data_postproc(py)

    # create a directory
    regr_dir = model_dir / f'regr_{voltage}Vpp_{pressure}Pa'
    if not os.path.exists(regr_dir):
        os.mkdir(regr_dir)
    
    # back up
    shutil.copyfile(__file__, regr_dir / 'do_regr.py')
    
    # save results
    # save_pred_vals(tX, py, rslt_dir=regr_dir) # values (csv)
    
    # create triangulations for tricontourf
    if on_grid:
        triangles = None  # TODO
    else:
        data_plot.difference_plot2(avg_data.iloc[:,:4], py, ty, regr_dir)
        # data_plot.all_plot(avg_data.iloc[:,:4], py, ty, regr_dir)
        # data_plot.all_plot(avg_data.iloc[:,:4], py, ty, regr_dir, simulation=True)  # plot simulation as reference
        triangles = data_plot.triangulate(pd.concat([avg_data.iloc[:,:4],py], axis='columns'))
        data_plot.quickplot(py, regr_dir, triangles=triangles, mesh=False)
        data_plot.quickplot(py, regr_dir, mesh=True, nodes=avg_data.iloc[:,2:4]*100)
    
    for n,p_param in enumerate(ty.columns, start=1): # figs
        if on_grid:
            filename = 'regr_fig_{0:02d}.png'.format(n)
        else:
            filename = 'regr_fig_{0:02d}_mesh.png'.format(n)
        fig_file = regr_dir / filename
        plot_df = pd.concat([avg_data.iloc[:,:4], py], axis='columns')

        data_plot.draw_a_2D_graph(plot_df, p_param, triangles, file_path=fig_file, 
                                  lin=lin, on_grid=on_grid, X_mesh=X, Y_mesh=Y)
    
    # scores if data available
    if ty.isnull().values.sum()==0:
        print()
        scores = print_scores(ty, py, regr_dir)
    
    if not on_grid:
        data_plot.correlation_torch(prediction=py, targets=ty, scores=scores, out_dir=regr_dir)

    d = datetime.datetime.today()
    print('finished on', d.strftime('%Y-%m-%d %H:%M:%S'))
