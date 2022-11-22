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
from tensorflow import keras
from pathlib import Path
import numpy as np

from sklearn.preprocessing import MinMaxScaler

import data


def create_output_dir():
    rslt_dir = Path('./created_models')
    if not os.path.exists(rslt_dir):
        os.mkdir(rslt_dir)

    date_str = datetime.datetime.today().strftime('%Y-%m-%d_%H%M')
    out_dir = rslt_dir + '/' + date_str
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    print('directory', out_dir, 'has been created.\n')

    return out_dir


def create_dataset(df):
    # split into x and y
    features = df[feature_names]  # TODO + x**2 and y**2
    labels = df[label_names]

    # scale x and y, save scalers
    sX = scale_all(features, 'x', out_dir=scaler_dir)
    sy = scale_all(labels, 'y', out_dir=scaler_dir)

    return tf.data.Dataset.from_tensor_slices((sX, sy))\
        .shuffle(len(sy)).batch(32)


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
    
    neurons = 512
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
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
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


################################################################


if __name__ == '__main__':
    # variables
    ks = 8
    feature_names = ['Vpp [V]', 'P [Pa]', 'X', 'Y']
    label_names = ['potential (V)', 'Ne (#/m^-3)', 'Ar+ (#/m^-3)', 'Nm (#/m^-3)',
                   'Te (eV)'] 
    
    voltages  = [200, 300, 400, 500] # V
    pressures = [  5,  10,  30,  45, 60, 80, 100, 120] # Pa
    powV, powP, powX, powY = 1, 1, 2, 2

    # paths
    work_dir = Path(os.getcwd())
    scaler_dir = Path()
    data_fldr_path = Path()
    out_dir = create_output_dir()
    
    # import data
    df = data.read_all_data(data_fldr_path, voltages, pressures)\
        .drop(columns=['Ex (V/m)','Ey (V/m)'])
        
    testV = 300
    testP = 60
    df.drop((df['Vpp [V]'] == testV) & (df['P [Pa]'] == testP)], inplace=True)
    
    # valV = []
    # valP = []
    # pairs = list(zip(valV, valP))
    
    now = dt.datetime.now().strftime('%Y-%m-%d_%H:%M')
    superdir = work_dir/ f'created_models/k-cross_{now}'
    if not os.path.exists(superdir):
        os.mkdir(work_dir / 'created_models' / superdir)
    
    for k in range(ks):
        val_df = df[    (df['Vpp [V]'] == pairs[k][0])
                    &   (df['Vpp [V]'] == pairs[k][1])]
        train_df = val_df # TODO
        
        train_df.to_csv()
        val_df.to_csv()
        
        x = create_dataset(train_df)
        y = create_dataset(val_df)
        
        num_descriptors = powV + powP + powX + powY
        num_obj_vars = len(df.columns) - 4
        
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
        
        model = create_model(num_descriptors, num_obj_vars)
        history = model.fit(df, epochs=100, verbose=1, callbacks=[early_stop])
        
        # save the model
        subdir = superdir / f'model_{k}-{ks}/' # TODO
        if not os.path.exists(subdir):
            os.mkdir(subdir)
    
        model.save(out_dir / 'model')
        print('NN model has been saved.\n')
    