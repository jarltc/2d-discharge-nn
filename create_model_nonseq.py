#coding: utf-8

''' model creation code created by Ikuse-san, modified by Jarl'''

import os, sys
import time, datetime
import pickle
import posixpath
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import mathtext

import sklearn
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

from tensorflow import keras
from tensorflow.keras.utils import plot_model
from tensorflow.python.client import device_lib

import data


def create_output_dir():
    rslt_dir = './created_models'
    if not os.path.exists(rslt_dir):
        os.mkdir(rslt_dir)
    
    date_str = datetime.datetime.today().strftime('%Y-%m-%d_%H%M')
    out_dir = rslt_dir + '/' + date_str
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    print('directory', out_dir, 'has been created.\n')
    
    return out_dir


def data_preproc(data_table):
    trgt_params = ('potential (V)', 'Ne (#/m^-3)', 'Ar+ (#/m^-3)', 'Nm (#/m^-3)', 'Te (eV)')
    for col_n,(col_name,col_vals) in enumerate(data_table.iteritems(), start=1):
        if col_name in trgt_params:
            tmp_col = np.log10(col_vals.values.reshape(-1,1))
        else:
            tmp_col = col_vals.values.reshape(-1,1)
        proced_table = tmp_col if col_n==1 else np.hstack([proced_table,tmp_col])
    
    proced_table = pd.DataFrame(proced_table, columns=data_table.columns)
    proced_table = proced_table.replace([np.inf,-np.inf], np.nan)
    proced_table = proced_table.dropna(how='any')
    
    return proced_table


def create_descriptor_table(data_table, pows, out_dir=None):
    pow_labels = ('V', 'P', 'x', 'y')
    dsc_linlog = []
    dsc_columns = []
    for col_n,(col_name,col_vals) in enumerate(data_table.iteritems()):
        pow_label = pow_labels[col_n]
        for pow_n in range(1,pows[pow_label]+1):
            tmp_col = col_vals.values.reshape(-1,1)**pow_n
            
            if np.log10(np.abs(tmp_col.max()-tmp_col.min()))>4:
                tmp_col = np.log10(tmp_col)
                dsc_linlog.append('log')
            else:
                dsc_linlog.append('lin')
            
            dsc_table = tmp_col if col_n==0 and pow_n==1 else np.hstack([dsc_table,tmp_col])
            
            label = pow_label if pow_n==1 else pow_label+'**{0:d}'.format(pow_n)
            dsc_columns.append(label)
    
    if out_dir is not None:
        with open(posixpath.join(out_dir,'dsc_linlog.pkl'),'wb') as pf:
            pickle.dump(dsc_linlog, pf, protocol=4)
        with open(posixpath.join(out_dir,'dsc_pows.pkl'),'wb') as pf:
            pickle.dump(pows, pf, protocol=4)
    
    return pd.DataFrame(dsc_table, columns=dsc_columns)


def scale_all(data_table, x_or_y, out_dir=None):
    for n,column in enumerate(data_table.columns, start=1):
        scaler = MinMaxScaler()
        data_col = data_table[column].values.reshape(-1,1)
        scaler.fit(data_col)
        scaled_data = scaler.transform(data_col)
        scaled_data_table = scaled_data if n==1 else np.hstack([scaled_data_table,scaled_data])
        
        if out_dir is not None:
            pickle_file = posixpath.join(out_dir, '{0:s}scaler_{1:02d}.pkl'.format(x_or_y,n))
            with open(pickle_file, mode='wb') as pf:
                pickle.dump(scaler, pf, protocol=4)
    
    scaled_data_table = pd.DataFrame(scaled_data_table, columns=data_table.columns)
    
    return scaled_data_table

def create_model(num_descriptors, num_obj_vars):
    '''
    Create the model and compile it

    Parameters
    ----------
    num_descriptors : int
        Size of input parameters
    num_obj_vars : int
        Size of output parameters

    Returns
    -------
    model : Keras model
        Model used to predict 2D discharges.

    '''
    neurons = 64
    layers = 10
    
    # model specification
    inputs = keras.Input(shape=(num_descriptors,))
    
    dense = keras.layers.Dense(neurons, activation='relu')
    x = dense(inputs) # first hidden layer
    
    # add additional hidden layers
    for i in range(layers-1):
        x = keras.layers.Dense(neurons, activation='relu')(x)
    
    outputs = keras.layers.Dense(num_obj_vars)(x)
    
    # build the model
    model = keras.Model(inputs=inputs, outputs=outputs, name='2d-nn_model')
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # compile the model
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    
    return model


def save_history_vals(history, out_dir):
    history_path = posixpath.join(out_dir, 'history.csv')
    history_table = np.hstack([np.array(history.epoch              ).reshape(-1,1),
                               np.array(history.history['mae']     ).reshape(-1,1),
                               np.array(history.history['val_mae'] ).reshape(-1,1),
                               np.array(history.history['loss']    ).reshape(-1,1),
                               np.array(history.history['val_loss']).reshape(-1,1)])
    history_df = pd.DataFrame(history_table, columns=['epoch','mae','val_mae','loss','val_loss'])
    history_df.to_csv(history_path, index=False)


def save_history_graph(history, out_dir, param='mae'):
    x  = np.array(history.epoch)
    if param=='mae':
        y1 = np.array(history.history['mae'])
        y2 = np.array(history.history['val_mae'])
    elif param=='loss':
        y1 = np.array(history.history['loss'])
        y2 = np.array(history.history['val_loss'])
    
    # matplotlib settings
    # mathtext.FontConstantsBase = mathtext.ComputerModernFontConstants
    # mathtext.FontConstantsBase.script_space = 0.01
    # mathtext.FontConstantsBase.delta = 0.01
    plt.rcParams['font.size'] = 12 # 12 is the default size
    # plt.rcParams['font.family'] = 'serif'
    # plt.rcParams['mathtext.default'] = 'default'
    # plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['xtick.minor.size'] = 2
    plt.rcParams['ytick.minor.size'] = 2
    fig = plt.figure(figsize=(6.0,6.0))
    
    # axis labels
    plt.xlabel('Epoch')
    if param=='mae':
        plt.ylabel('MAE')
    elif param=='loss':
        plt.ylabel('Loss')
    
    plt.plot(x, y1, color='green', lw=1.0, label='train')
    plt.plot(x, y2, color='red'  , lw=1.0, label='test')
    
    # set both x_min and y_min as zero
    ax = plt.gca()
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    
    ax.xaxis.get_ticklocs(minor=True)
    ax.yaxis.get_ticklocs(minor=True)
    ax.minorticks_on()
    
    plt.legend()
    plt.grid()
    plt.tight_layout()
    
    # save the figure
    if param=='mae':
        file_name = 'history_graph_mae.png'
    elif param=='loss':
        file_name = 'history_graph_loss.png'
    file_path = posixpath.join(out_dir, file_name)
    fig.savefig(file_path)


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
    device_lib.list_local_devices()
    print()
    
    # -------------------------------------------------------
    
    voltages  = [200, 300, 400, 500] # V
    pressures = [  5,  10,  30,  45, 60, 80, 100, 120] # Pa
    
    data_fldr_path = './data/avg_data'
    
    voltage_excluded = 300 # V
    pressure_excluded = 60 # Pa
    
    powV, powP, powX, powY = 1, 1, 2, 2
    
    # -------------------------------------------------------
    
    out_dir = create_output_dir()
    
    # copy some files for backup
    
    shutil.copyfile(posixpath.join('.',sys.argv[0]), posixpath.join(out_dir, 'create_model.py'))
    shutil.copyfile(posixpath.join('.','data.py'), posixpath.join(out_dir,'data.py'))
    
    # get dataset
    print('start getting dataset...', end='', flush=True)
    start_time = time.time()
    avg_data = data.read_all_data(data_fldr_path, voltages, pressures)
    if len(avg_data)==0:
        print('error: no data available.')
        raise Exception
    elapsed_time = time.time() - start_time
    print(' done ({0:.1f} sec).\n'.format(elapsed_time))
    #print(avg_data)
    
    # prepare data
    avg_data = avg_data.drop(columns=['Ex (V/m)','Ey (V/m)'])
    data_used     = avg_data[~((avg_data['Vpp [V]']==voltage_excluded) & (avg_data['P [Pa]']==pressure_excluded))]
    data_excluded = avg_data[  (avg_data['Vpp [V]']==voltage_excluded) & (avg_data['P [Pa]']==pressure_excluded) ]
    print('start preproc...', flush=True)
    pows = {'V':powV, 'P':powP, 'x':powX, 'y':powY}
    descriptors = create_descriptor_table(data_used.iloc[:,:4], pows, out_dir)
    train_data = data_preproc(pd.concat([descriptors,data_used.iloc[:,4:]], axis=1))
    print('done.\n')
    
    num_descriptors = powV + powP + powX + powY
    num_obj_vars = len(avg_data.columns) - 4
    print('num data points =', len(train_data))
    print('num descriptors =', num_descriptors)
    print('num obj vars    =', num_obj_vars)
    print()
    
    # randomly permutate
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    
    # store data for backup
    train_data.to_csv(posixpath.join(out_dir,'data_used.csv'), index=False)
    data_excluded.to_csv(posixpath.join(out_dir,'data_excluded.csv'), index=False)
    
    # scale all data by MinMaxScaler
    scaler_dir = out_dir + '/scalers'
    os.mkdir(scaler_dir)
    sX = scale_all(train_data.iloc[:,:num_descriptors], 'x', out_dir=scaler_dir)
    sy = scale_all(train_data.iloc[:,num_descriptors:], 'y', out_dir=scaler_dir)

    sX = tf.convert_to_tensor(sX)
    sy = tf.convert_to_tensor(sy)
    
    ''' additional preprocessing:
        * normalization layer after taking the log
        * shuffling the data    
    '''
        
    
    # --------
    
    # create a regression model
    print('start creating a model...', flush=True)
    
    model = create_model(num_descriptors, num_obj_vars)
    model.summary()
    
    # the patience parameter is the amount of epochs to check for improvement.
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
     
    # store training stats
    history = model.fit(sX, sy, epochs=100, validation_split=0.1, 
                        verbose=1, callbacks=[early_stop])
    #history = model.fit(sX, sy, epochs=500, validation_split=0.1, verbose=2)
    
    print('\ndone.\n', flush=True)
    
    # --------
    
    # create a figure file of NN model structure if possible
    try:
        plot_model(
            model, to_file=posixpath.join(out_dir,'model_structure.png'),
            show_shapes=True, show_layer_names=False)
        print('figure of the NN model structure has been saved.\n')
    except:
        pass
    
    # save the NN model
    model.save(out_dir+'/model')
    print('NN model has been saved.\n')
    
    #print(list(history.history.keys()))
    save_history_vals(history, out_dir)
    save_history_graph(history, out_dir, 'mae')
    save_history_graph(history, out_dir, 'loss')
    print('NN training history has been saved.\n')
    
    d = datetime.datetime.today()
    print('finished on', d.strftime('%Y-%m-%d %H:%M:%S'))

    os.system("notify-send \"Job finished\"!")
