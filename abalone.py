#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict abalone age from shell measurements
Created on Wed Nov 16 21:53:38 2022

@author: jarl
"""

import pandas as pd
import numpy as np

np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
# tf.config.set_visible_devices([],'GPU')

from tensorflow.keras import layers

url = "https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv"
abalone_train = pd.read_csv(url, names=["Length", "Diameter", "Height", 
                                        "Whole weight", "Shucked weight",
                                        "Viscera weight", "Shell weight", 
                                        "Age"])

# predict age from other measurements

abalone_features = abalone_train.copy()
abalone_labels = abalone_features.pop('Age')

# treat all features identically, pack into a single NumPy array:
abalone_features = np.array(abalone_features)

# since there is only a single input tensor, a tf.keras.Sequential model is sufficient
abalone_model = tf.keras.Sequential([
    layers.Dense(64),
    layers.Dense(1)
    ])

abalone_model.compile(loss = tf.keras.losses.MeanSquaredError(),
                      optimizer = tf.keras.optimizers.Adam())

abalone_model.fit(abalone_features, abalone_labels, epochs=10)

###### model works up to here ######

# it's good practice to normalize the inputs to the model
normalize = layers.Normalization()

# use Normalization.adapt method to adapt the layer to the data
normalize.adapt(abalone_features)

# use the layer in the model
norm_abalone_model = tf.keras.Sequential([
    normalize,
    layers.Dense(64),
    layers.Dense(1)
    ])

norm_abalone_model.compile(loss = tf.keras.losses.MeanSquaredError(),
                           optimizer = tf.keras.optimizers.Adam())

norm_abalone_model.fit(abalone_features, abalone_labels, epochs=10)
