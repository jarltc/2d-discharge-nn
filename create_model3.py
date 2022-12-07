"""
Load data and train a model. (version 3)

6 Dec 13:30
author @jarl
"""

import os
import sys
import time
import shutil
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import tensorflow as tf
import datetime
from tensorflow import keras
import posixpath
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import data

tf.config.set_visible_devices([], 'GPU')

def create_output_dir():
    rslt_dir = Path('./created_models')
    if not os.path.exists(rslt_dir):
        os.mkdir(rslt_dir)
    
    date_str = datetime.datetime.today().strftime('%Y-%m-%d_%H%M')
    out_dir = rslt_dir / date_str
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    print('directory', out_dir, 'has been created.\n')
    
    return out_dir


def data_preproc(data_table, lin=True):
    global scale_exp
    trgt_params = ('potential (V)', 'Ne (#/m^-3)', 'Ar+ (#/m^-3)', 'Nm (#/m^-3)', 'Te (eV)')

    for col_n,(col_name,col_vals) in enumerate(data_table.iteritems(), start=1):
        if col_name in trgt_params:
            if lin:
                exponent = round(np.log10(col_vals.mean()), 0) - 1.0  # get exponent for scaling
                scale_exp.append(exponent)            
                tmp_col = col_vals.values.reshape(-1,1)/(10**exponent)  # scale by dividing
            else:
                tmp_col = np.log10(col_vals.values.reshape(-1,1))
        else:
            tmp_col = col_vals.values.reshape(-1,1)
        proced_table = tmp_col if col_n==1 else np.hstack([proced_table,tmp_col])
    
    proced_table = pd.DataFrame(proced_table, columns=data_table.columns)
    proced_table = proced_table.replace([np.inf,-np.inf], np.nan)
    proced_table = proced_table.dropna(how='any')
    
    return proced_table


def scale_all(data_table, x_or_y, out_dir=None):
    data_table = data_table.copy()
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
    global neurons, layers, learning_rate, name
    
    # model specification
    inputs = keras.Input(shape=(num_descriptors,))
    
    dense = keras.layers.Dense(neurons, activation='relu')
    x = dense(inputs) # first hidden layer
    
    # add additional hidden layers
    for i in range(layers-1):
        x = keras.layers.Dense(neurons, activation='relu')(x)
    
    outputs = keras.layers.Dense(num_obj_vars)(x)
    
    # build the model
    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # compile the model
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    
    return model


def save_history_graph(history, out_dir, param='mae'):
    matplotlib.rcParams['font.family'] = 'Arial'
    x  = np.array(history.epoch)
    if param=='mae':
        y1 = np.array(history.history['mae'])
        y2 = np.array(history.history['val_mae'])
    elif param=='loss':
        y1 = np.array(history.history['loss'])
        y2 = np.array(history.history['val_loss'])
    

    plt.rcParams['font.size'] = 12 # 12 is the default size
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


def save_history_vals(history, out_dir):
    history_path = posixpath.join(out_dir, 'history.csv')
    history_table = np.hstack([np.array(history.epoch              ).reshape(-1,1),
                               np.array(history.history['mae']     ).reshape(-1,1),
                               np.array(history.history['val_mae'] ).reshape(-1,1),
                               np.array(history.history['loss']    ).reshape(-1,1),
                               np.array(history.history['val_loss']).reshape(-1,1)])
    history_df = pd.DataFrame(history_table, columns=['epoch','mae','val_mae','loss','val_loss'])
    history_df.to_csv(history_path, index=False)


def yn(str):
    if str.lower() in ['y', 'yes', 'ok', 'sure', 'hai']:
        return True
    elif str.lower() in ['n', 'no', 'nope', 'nah', 'hold this L']:
        return False
    else:
        raise Exception(str + 'not recognized: use y - yes, n - no')


# --------------- Model hyperparameters -----------------
# model name
name = input('Enter model name: ')

# training
batch_size = int(input('Batch size (default 32): ') or '32')
learning_rate = float(input('Learning rate (default 0.001): ') or '0.001')
validation_split = float(input('Validation split (default 0.1): ') or '0.1')
epochs = int(input('Epochs (default 100): ') or '100')
minmax_y = yn(input('Scale target data? [y/n]: '))
lin = yn(input('Scale output data linearly? [y/n]: '))

# architecture
neurons = 64
layers = 10

# -------------------------------------------------------

voltages  = [200, 300, 400, 500] # V
pressures = [  5,  10,  30,  45, 60, 80, 100, 120] # Pa

data_fldr_path = './data/avg_data'

voltage_excluded = 300 # V
pressure_excluded = 60 # Pa

# -------------------------------------------------------

out_dir = create_output_dir()
scaler_dir = out_dir / 'scalers'
os.mkdir(scaler_dir)

# copy some files for backup
shutil.copyfile(posixpath.join('.',sys.argv[0]), posixpath.join(out_dir, 'create_model.py'))
shutil.copyfile(posixpath.join('.','data.py'), posixpath.join(out_dir,'data.py'))

# get dataset
print('start getting dataset...', end='', flush=True)
start_time = time.time()
avg_data = data.read_all_data(data_fldr_path, voltages, pressures).drop(columns=['Ex (V/m)', 'Ey (V/m)'])
if len(avg_data)==0:
    print('error: no data available.')
    raise Exception
elapsed_time = time.time() - start_time
print(f' done ({elapsed_time:0.1f} sec).\n')

# separate data to be excluded (to later check the model)
data_used     = avg_data[~((avg_data['Vpp [V]']==voltage_excluded) & (avg_data['P [Pa]']==pressure_excluded))].copy()
data_excluded = avg_data[  (avg_data['Vpp [V]']==voltage_excluded) & (avg_data['P [Pa]']==pressure_excluded) ].copy()

feature_names = ['V', 'P', 'x', 'x**2', 'y', 'y**2']
label_names = ['potential (V)', 'Ne (#/m^-3)', 'Ar+ (#/m^-3)', 'Nm (#/m^-3)', 'Te (eV)']

# create descriptor table from features
data_used.rename(columns={'Vpp [V]' : 'V',
                          'P [Pa]'  : 'P',
                          'X'       : 'x', 
                          'Y'       : 'y'}, inplace=True)

data_used['x**2'] = data_used['x']**2
data_used['y**2'] = data_used['y']**2

# scale features and labels
scale_exp = []

features = scale_all(data_used[feature_names], 'x', scaler_dir).astype('float64')
labels = data_preproc(data_used[label_names]).astype('float64')

if minmax_y:
    labels = scale_all(labels, 'y', scaler_dir)

alldf = pd.concat([features, labels], axis=1)
dataset_size = len(alldf)

# save the data
alldf.to_csv(out_dir / 'data_used.csv', index=False) 
data_excluded.to_csv(out_dir / 'data_excluded.csv', index=False)

# create tf dataset object and shuffle it
dataset = tf.data.Dataset.from_tensor_slices((features, labels)).shuffle(dataset_size)

# determine validation split
train_size = int((1-validation_split) * dataset_size)
val_size = int(validation_split * dataset_size)

# create validation split
train_ds = dataset.take(train_size).batch(batch_size)
val_ds = dataset.skip(train_size).take(val_size).batch(batch_size)

model = create_model(len(feature_names), len(label_names))
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)

print('begin model training...')
history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=[early_stop])
print('\ndone.\n', flush=True)

model.save(out_dir / 'model')
print('NN model has been saved.\n')

save_history_vals(history, out_dir)
save_history_graph(history, out_dir, 'mae')
save_history_graph(history, out_dir, 'loss')
print('NN training history has been saved.\n')

# save metadata
metadata = {'name' : name,  # str
            'scaling' : lin,  # bool
            'is_target_scaled': minmax_y,  # bool
            'parameter_exponents': scale_exp}  # list of float

with open('train_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

d = datetime.datetime.today()
print('finished on', d.strftime('%Y-%m-%d %H:%M:%S'))
