#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code to create a model and cross-validate with leave-one-out (LOOCV)
Created on Tue Nov 22 13:22:28 2022

@author: jarl
"""

import sys, os
import posixpath
import pickle
import datetime as dt

import pandas as pd
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from tensorflow import keras
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold

from sklearn.preprocessing import MinMaxScaler

import data


def create_output_dir():
    rslt_dir = Path('./created_models')
    if not os.path.exists(rslt_dir):
        os.mkdir(rslt_dir)

    date_str = dt.datetime.today().strftime('%Y-%m-%d_%H%M')
    out_dir = rslt_dir / date_str
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    print('directory', out_dir, 'has been created.\n')

    return out_dir


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


def create_dataset(df, tensor=False):
    # split into x and y
    global scaler_dir
    pows = {'V':powV, 'P':powP, 'x':powX, 'y':powY}
    features = create_descriptor_table(df.iloc[:,:4], pows, out_dir)
    labels = df[label_names]

    # scale x and y, save scalers
    sX = scale_all(features, 'x', out_dir=scaler_dir)
    sy = scale_all(labels, 'y', out_dir=scaler_dir)
    
    if tensor==True:
        return tf.data.Dataset.from_tensor_slices((sX, sy))
        #\.shuffle(len(sy)).batch(32)
    else: return(sX, sy)


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
    layers = 4
    
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
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
    
    # compile the model
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    
    return model


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


def draw_line():
    term_size = os.get_terminal_size()
    print('-' * term_size.columns)
    

################################################################


if __name__ == '__main__':
    # variables
    num_folds = 6
    feature_names = ['Vpp [V]', 'P [Pa]', 'X', 'Y']
    label_names = ['potential (V)', 'Ne (#/m^-3)', 'Ar+ (#/m^-3)', 'Nm (#/m^-3)',
                   'Te (eV)'] 
    
    voltages  = [200, 300, 400, 500] # V
    pressures = [  5,  10,  30,  45, 60, 80, 100, 120] # Pa
    powV, powP, powX, powY = 1, 1, 2, 2

    # paths
    work_dir = Path(os.getcwd())
    data_fldr_path = './data/avg_data'
    out_dir = create_output_dir()
    ##### print when testing ###
    with open(Path(out_dir)/'this_is_a_test.txt', 'w') as f:
        f.write('thanks!')
        
    scaler_dir = Path(out_dir) / 'scalers'
    os.mkdir(scaler_dir)
    
    # import data
    df = data.read_all_data(data_fldr_path, voltages, pressures)\
        .drop(columns=['Ex (V/m)','Ey (V/m)'])
    
    num_descriptors = powV + powP + powX + powY
    num_obj_vars = len(df.columns) - 4
    
    # split into train and test (not val)
    testV = 300
    testP = 60
    test_df = df.loc[(df['Vpp [V]'] == testV) & (df['P [Pa]'] == testP)]
    train_df = df.drop(test_df.index).reset_index(drop=True)
    
    train_df.to_csv(Path(out_dir)/'data_used.csv', index=False)
    test_df.to_csv(Path(out_dir)/'data_excluded.csv', index=False)
    
    x, y = create_dataset(train_df)
    z = create_dataset(test_df, tensor=True)
    mse_per_fold    = []
    mae_per_fold    = []
    r2_per_fold     = []
    
    sys.exit()  # run code up to here
    
    kfold = KFold(n_splits=num_folds, shuffle=True)
    fold_no = 1
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
    
    for train, val in kfold.split(x, y):
        model = create_model(num_descriptors, num_obj_vars)
        
        draw_line()
        print(f'\nTraining for fold {fold_no}...')
        
        history = model.fit(x, y, epochs=100, verbose=1, callbacks=[early_stop])
        print('\nComplete!\n')
        
        draw_line()
        print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}')
        fold_no += 1
        
    for k in range(ks):
    
        model.save(out_dir / 'model')
        print('NN model has been saved.\n')
    