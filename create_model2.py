#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model creation v2
Create neural network models using TensorFlow input pipelines

Created on Wed Nov 16 11:32:32 2022

@author: jarl
"""

import tensorflow as tf
import os, sys
import pandas as pd
from pathlib import Path
import data
import datetime
import numpy as np
import matplotlib.pyplot as plt

tf.config.set_visible_devices([], 'GPU')

def get_basic_model():
  model = tf.keras.Sequential([
    normalizer,
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(5)
  ])

  optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
  
  model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
  return model


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


def save_history_graph(history, param='mae'):
    x  = np.array(history.epoch)
    if param=='mae':
        y1 = np.array(history.history['mae'])
        # y2 = np.array(history.history['val_mae'])
    elif param=='loss':
        y1 = np.array(history.history['loss'])
        # y2 = np.array(history.history['val_loss'])
    
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
    # plt.plot(x, y2, color='red'  , lw=1.0, label='test')
    
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


data_fldr_path = Path('./data/avg_data')
out_dir = create_output_dir()

voltages    =   [200, 300, 400, 500] #V
pressures   =   [  5,  10,  30,  45, 60, 80, 100, 120]# Pa

print('reading data...\n')
df = data.read_all_data(data_fldr_path, voltages, pressures)
df = df.drop(columns=['Ex (V/m)', 'Ey (V/m)'])
df = df.astype({'Vpp [V]':'float64', 'P [Pa]':'float64'})
df = df[(df['Vpp [V]'] == 300) & (df['P [Pa]'] == 60)]
print('complete!\n')

feature_names = ['Vpp [V]', 'P [Pa]', 'X', 'Y']
features = df[feature_names]

label_names = ['potential (V)', 'Ne (#/m^-3)', 'Ar+ (#/m^-3)', 'Nm (#/m^-3)',
               'Te (eV)']
labels = df[label_names]
# labels.iloc[:,1:-1] = np.log(labels.iloc[:,1:-1])

sys.exit()

avg_data = tf.data.Dataset.from_tensor_slices((features, labels))

batches = avg_data.shuffle(1000).batch(32)

model = get_basic_model()
model.summary()
history = model.fit(batches, epochs=5, verbose=1)

save_history_graph(history, 'loss')

model.save(out_dir/'model')
print('NN model has been saved.\n')

                