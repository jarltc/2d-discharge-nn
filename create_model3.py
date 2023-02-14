"""
Load data and train a model. (version 3)

6 Dec 13:30
author @jarl
"""

import os
import sys
import time
import data
import shutil
# from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pickle
import datetime
from pathlib import Path


import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf
from scipy.spatial import cKDTree
from tensorflow import keras
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

tf.config.set_visible_devices([], 'GPU')  # disable gpu (cpu is faster)

# arguments
# parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
# parser.add_argument('-n', '--name', default=None, help='Model name.')
# parser.add_argument('-l', '--log', action='store_true', help='Scale data logarithmically.')
# parser.add_argument('-u', '--unscaleY', action='store_true', help='Leave target variables unscaled.')
# parser.add_argument('-y', '--layers', default=10, help='Specify layer count.')
# parser.add_argument('-o', '--nodes', default=64, help='Specify nodes per layer.')
# args = vars(parser.parse_args())

def create_output_dir():
    rslt_dir = root / 'created_models'
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

    def get_param_exp(col_vals):
        ''' get exponent of the parameter's mean value for scaling. '''
        mean_exp = round(np.log10(col_vals.mean()), 0) - 1.0
        # return 0 if mean_exp is less than zero to avoid blowing up small values
        if mean_exp >=  0.0:
            return mean_exp
        else:
            return 0.0


    for col_n,(col_name,col_vals) in enumerate(data_table.iteritems(), start=1):
        if col_name in trgt_params:
            if lin:
                # get exponent for scaling
                exponent = get_param_exp(col_vals)
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
            pickle_file = out_dir / f'{x_or_y}scaler_{n:02d}.pkl'
            with open(pickle_file, mode='wb') as pf:
                pickle.dump(scaler, pf, protocol=4)
    
    scaled_data_table = pd.DataFrame(scaled_data_table, columns=data_table.columns)
    
    return scaled_data_table


def create_model_old(num_descriptors, num_obj_vars):
    '''
    Create the model and compile it (not optimized)

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


def create_model(num_descriptors, num_obj_vars):
    """Create a model.

    Model layers, activation,

    Args:
        num_descriptors (_type_): _description_
        num_obj_vars (_type_): _description_

    Returns:
        _type_: _description_
    """
    weight_decay = 7.480215373453682e-10  # TODO: optimize value for (loss - val_loss)
    def hidden_layer(neurons):
      return keras.layers.Dense(neurons, activation=tf.nn.relu)

    weight_decay = 7.480215373453682e-10

    inputs = keras.Input(shape=(num_descriptors,))

    # hidden layers
    x = keras.layers.Dense(116, activation=tf.nn.relu, 
                           input_shape=(num_descriptors,))(inputs)
    x = hidden_layer(115)(x)
    x = hidden_layer(78)(x)
    x = hidden_layer(26)(x)
    x = hidden_layer(46)(x)
    x = hidden_layer(82)(x)
    x = hidden_layer(106)(x)

    outputs = keras.layers.Dense(num_obj_vars)(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    
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
    file_path = out_dir / file_name
    fig.savefig(file_path)


def save_history_vals(history, out_dir):
    history_path = out_dir / 'history.csv'
    history_table = np.hstack([np.array(history.epoch              ).reshape(-1,1),
                               np.array(history.history['mae']     ).reshape(-1,1),
                               np.array(history.history['val_mae'] ).reshape(-1,1),
                               np.array(history.history['loss']    ).reshape(-1,1),
                               np.array(history.history['val_loss']).reshape(-1,1)])
    history_df = pd.DataFrame(history_table, columns=['epoch','mae','val_mae','loss','val_loss'])
    history_df.to_csv(history_path, index=False)


def yn(str):
    if str.lower() in ['y', 'yes', 'yea', 'ok', 'okay', 'k',  
                       'sure', 'hai', 'aye', 'ayt', 'fosho']:
        return True
    elif str.lower() in ['n', 'no', 'nope', 'nah', 'hold this l']:
        return False
    else:
        raise Exception(str + 'not recognized: use y - yes, n - no')


##### data processing ######
def read_aug_data(file):
    """Read data file and return a DataFrame.

    Args:
        file (PosixPath): Path to .feather file.

    Returns:
        interp_df: DataFrame of interpolated data.
    """
    interp_df = pd.read_feather(file).drop(columns=['Ex (V/m)', 'Ey (V/m)'])
    return interp_df


def get_augmentation_data(data_used, xy: bool, vp: bool):
    """Get augmentation data for training.

    Args:
        xy (bool): Include xy grid augmented data
        vp (bool): Inclde vp augmentation data
    """
    if xy:  # xy augmentation
        xyfile = Path(root/'data'/'interpolation_datasets'/'rec-interpolation2.nc')
        xydf = xr.open_dataset(xyfile).to_dataframe().reset_index().dropna()
    else: xydf = None

    if vp:  # vp augmentation
        vpfolder = Path(root/'data'/'interpolation_feather'/'20221209')
        # read all files and combine into a single df
        vpdf = [read_aug_data(file) for file in vpfolder.glob('*.feather')]
        vpdf = pd.concat(vpdf).rename(columns={'Vpp [V]' : 'V', 
                                               'P [Pa]'  : 'P',
                                               'X'       : 'x',
                                               'Y'       : 'y'}, inplace=True)
    else: vpdf = None

    # make sure that the data follows the correct format before returning
    assert list(data_used.columns) == ['V', 'P', 'x', 'y'] + label_names
    return pd.concat([data_used, xydf, vpdf], ignore_index=True)
    

def get_data(xy=False, vp=False):
    """Get dataset

    Assumes the DataFrame has previously been saved as a .feather file. If not,
    a new .feather file is created in the root/data folder.

    Args:
        xy (bool, optional): Include xy augmentation. Defaults to False.
        vp (bool, optional): Include vp augmentation. Defaults to False.

    Raises:
        Exception: Raises an error if no data is available in avg_data.

    Returns:
        data_used: DataFrame of data to be used, with specified augmentation data.
        data_excluded: DataFrame of excluded data.
    """
    avg_data_file = root/'data'/'avg_data.feather'

    # check if feather file exists and load avg_data
    if avg_data_file.is_file():
        print('reading data feather file...')
        avg_data = pd.read_feather(avg_data_file)
    
    else:
        print('data feather file not found, building file...')
        start_time = time.time()
        avg_data = data.read_all_data(data_fldr_path, voltages, pressures).drop(columns=['Ex (V/m)', 'Ey (V/m)'])
        elapsed_time = time.time() - start_time
        print(f' done ({elapsed_time:0.1f} sec).\n')

        if len(avg_data)==0:
            raise Exception('No data available.')

        avg_data.to_feather(avg_data_file)

    # separate data to be excluded (to later check the model)
    data_used     = avg_data[~((avg_data['Vpp [V]']==voltage_excluded) & (avg_data['P [Pa]']==pressure_excluded))].copy()
    data_excluded = avg_data[  (avg_data['Vpp [V]']==voltage_excluded) & (avg_data['P [Pa]']==pressure_excluded) ].copy()

    # rename columns
    data_used.rename(columns={'Vpp [V]' : 'V',
                              'P [Pa]'  : 'P',
                              'X'       : 'x', 
                              'Y'       : 'y'}, inplace=True)

    if (xy or vp):
        data_used = get_augmentation_data(data_used, xy, vp)

    # create new column of x^2 and y^2
    # data_used['x**2'] = data_used['x']**2
    # data_used['y**2'] = data_used['y']**2

    try: 
        data_used.drop(columns=['x**2', 'y**2'], inplace=True)
    except:
        print("Columns x**2 and y**2 don't exist lol")
    
    return data_used, data_excluded


###### neighbor regularization ######
def neighbor_mean(point, k):
    # get a point's neighbors and get a vector of the means for each variable (shape: (k,5))
    x, y, v, p = point  # unpack point, returns tensors
    v = np.array([v.numpy()])  # convert tensor to 1D vector (numpy)
    p = np.array([p.numpy()])

    _, ii = tree.query([x, y], k)  # get nearest k neighbors of the point
    
    # get pair of x, y of point's neighbors
    neighbor_xy = [features[['x', 'y']].iloc[i].to_numpy() for i in ii]

    # combine (x,y) with v and p, as input to the model
    neighbors = [np.concatenate((xy, v, p)) for xy in neighbor_xy]  # list of input vectors x

    # convert to tensor (expand dims cause it expects a batch size)
    neighbors = [tf.expand_dims(tf.convert_to_tensor(neighbor), axis=0) for neighbor in neighbors]
    
    # get mean of neighbor predictions for 5 variables
    return tf.reduce_mean([model(neighbor) for neighbor in neighbors], axis=0)


def neighbor_loss(x_batch, y_batch, training=False, k=3):
    """Calculate neighbor difference for each training point in a batch.

    This loss is added as a regularizer to improve smoothness.
    Args:
        x_batch (tf.Tensor): Batched features of the specified batch size.
        y_batch (tf.Tensor): Batched labels of the specified batch size.
        training (bool, optional): Make model weights trainable. Defaults to False.
        k (int, optional): Number of neighbors to query. Defaults to 4.

    Returns:
        Total MSE: tf.Tensor of shape(batch_size, ) specifying the MSE for each item in the batch.
    """
    global c
    y_pred = model(x_batch, training=training)
    
    neighbor_means = [neighbor_mean(point, k) for point in x_batch]    
    batch_mean = tf.concat(neighbor_means, axis=0)

    def neighbor_loss_core(y_true, y_pred):
        """Calculate the combined MSE.

        Args:
            y_true (tf.Tensor): Tensor of shape (batch_size, num_labels).
            y_pred (tf.Tensor): Output of model(x_batch), with a shape of (batch_size, num_labels)

        Returns:
            Total MSE: tf.Tensor of shape(batch_size, ) specifying the MSE for each item in the batch.
        """
        mse_train = tf.cast(tf.losses.mean_squared_error(y_pred, y_true), 'float64')
        mse_neighbor = tf.cast(c*tf.losses.mean_squared_error(batch_mean, y_pred), 'float64')

        return tf.math.add(mse_train, mse_neighbor)

    return neighbor_loss_core(y_pred, y_batch)


def c_e(epoch, c=0.5, r=25, which='exp'):
    """Get regularization coefficient following a specified curve.

    function c(e), where e is the epoch and c is the coefficient at the epoch.

    Args:
        epoch (int): Current epoch.
        c (float, optional): Regularization coefficient to be approached. Defaults to 0.5.
        r (int, optional): Rate of increase. Coefficient increases by {} over r epochs. Defaults to 25.

    Returns:
        c: Coefficient at each epoch.
    """

    if which=='exp':
        return c - c*np.exp(-epoch/r)
    elif which=='sigmoid':
        k = 0.085
        x_0 = 100
        return c/(1 + np.exp(-k*(x-x_0)))


# --------------- Model hyperparameters -----------------
# model name
# if args['name'] == None:
name = input('Enter model name: ')
# else:
#     name = args['name']

root = Path(os.getcwd())
data_fldr_path = root/'data'/'avg_data'

# user inputs
batch_size = int(input('Batch size (default 128): ') or '128')
learning_rate = float(input('Learning rate (default 0.001): ') or '0.001')
validation_split = float(input('Validation split (default 0.1): ') or '0.1')
epochs = int(input('Epochs (default 5): ') or '5')
xy = yn(input('Grid augment (y/n, default y): ') or 'y')
vp = yn(input('VP augment (y/n, default y): ') or 'y')
minmax_y = True  # opposite of args[unscaleY], i.e.: False if unscaleY flag is raised
lin = True  # opposite of args[log], i.e.: False if log flag is raised

# architecture
neurons = None # args['nodes']
layers = None # args['layers']

# -------------------------------------------------------

voltages  = [200, 300, 400, 500] # V
pressures = [  5,  10,  30,  45, 60, 80, 100, 120] # Pa


voltage_excluded = 300 # V
pressure_excluded = 60 # Pa

# -------------------------------------------------------

out_dir = root/'created_models'/'test_dir' if name == 'test' else create_output_dir()
scaler_dir = out_dir / 'scalers'
if not scaler_dir.is_dir():
    os.mkdir(scaler_dir) 

# copy some files for backup (probably made redundant by metadata)
shutil.copyfile(__file__, out_dir / 'create_model.py')
shutil.copyfile(root / 'data.py', out_dir / 'data.py')

# feature_names = ['V', 'P', 'x', 'x**2', 'y', 'y**2']
feature_names = ['V', 'P', 'x', 'y']
label_names = ['potential (V)', 'Ne (#/m^-3)', 'Ar+ (#/m^-3)', 'Nm (#/m^-3)', 'Te (eV)']

# get data
data_used, data_excluded = get_data(xy, vp)

# set threshold to make very small values zero
pd.set_option('display.chop_threshold', 1e-10)

# scale features and labels
scale_exp = []
features = scale_all(data_used[feature_names], 'x', scaler_dir).astype('float64')
labels = data_preproc(data_used[label_names]).astype('float64')

if minmax_y:  # if applying minmax to target data
    labels = scale_all(labels, 'y', scaler_dir)

alldf = pd.concat([features, labels], axis=1)
dataset_size = len(alldf)

# create tf dataset object and shuffle it
dataset = tf.data.Dataset.from_tensor_slices((features, labels)).shuffle(dataset_size)

# determine validation split
train_size = int((1-validation_split) * dataset_size)
val_size = int(validation_split * dataset_size)

# split and batch the data
train_ds = dataset.take(train_size).batch(batch_size)
val_ds = dataset.skip(train_size).take(val_size).batch(batch_size)

# if on mesh, create ckdtree of grid points
nodes_df = scale_all(data_excluded[['X', 'Y']], 'x') 
tree = cKDTree(np.c_[nodes_df['X'].to_numpy(), nodes_df['Y'].to_numpy()])

# create the model 
model = create_model(len(feature_names), len(label_names))
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
# model.compile(loss=, optimizer=optimizer, metrics=['mae'])

# callbacks
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

time_callback = TimeHistory()  # record time of each epoch

# use tensorboard to monitor training
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=out_dir, histogram_freq=1)

# train the model
print('begin model training...')
train_start = time.time()  # record start time
# history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=[tensorboard_callback, time_callback])  # trains the model

# custom training loop
from tqdm import tqdm
for epoch in tqdm(range(epochs)):
    print(f'Start epoch {epoch}')
    epoch_loss_avg = tf.keras.metrics.Mean()
    c = c_e(epoch)
    
    for (x, y) in tqdm(train_ds):
        # open a GradientTape to record operations
        with tf.GradientTape() as tape:
            loss_value = neighbor_loss(x, y, training=True)
        
        # compute the gradient of the loss with respect to the model's weights
        grads = tape.gradient(loss_value, model.trainable_weights)
        
        # use the optimizer to update model's weights
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        
        # update the epoch loss
        epoch_loss_avg.update_state(loss_value)

    # end epoch
    print(f"Epoch {epoch}: loss = {epoch_loss_avg.result()}")


print('\ndone.\n', flush=True)
train_end = time.time()  # record end time

# save the model
model.save(out_dir / 'model')
print('NN model has been saved.\n')

# save_history_vals(history, out_dir)
# save_history_graph(history, out_dir, 'mae')
# save_history_graph(history, out_dir, 'loss')
# print('NN training history has been saved.\n')

# save metadata
metadata = {'name' : name,  # str
            'scaling' : lin,  # bool
            'is_target_scaled': minmax_y,  # bool
            'parameter_exponents': scale_exp}  # list of float

with open(out_dir / 'train_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

# record time per epoch
times = time_callback.times
with open(out_dir / 'times.txt', 'w') as f:
    f.write('Train times per epoch\n')
    for i, time in enumerate(times):
        time = round(time, 2)
        f.write(f'Epoch {i+1}: {time} s\n')

d = datetime.datetime.today()
print('finished on', d.strftime('%Y-%m-%d %H:%M:%S'))

# human-readable metadata
with open(out_dir / 'train_metadata.txt', 'w') as f:
    f.write(f'Model name: {name}\n')
    f.write(f'Lin scaling: {lin}\n')
    f.write(f'Number of points: {len(data_used)}\n')
    f.write(f'Target scaling: {minmax_y}\n')
    f.write(f'Parameter exponents: {scale_exp}\n')
    f.write(f'Execution time: {(train_end-train_start):.2f} s\n')
    f.write(f'Average time per epoch: {np.array(times).mean():.2f} s\n')
    f.write(f'\nUser-specified hyperparameters\n')
    f.write(f'Batch size: {batch_size}\n')
    f.write(f'Learning rate: {learning_rate}\n')
    f.write(f'Validation split: {validation_split}\n')
    f.write(f'Epochs: {epochs}\n')
    f.write(f'Grid augment: {xy}\n')
    f.write(f'VP augment: {vp}\n')
    f.write('\n*** end of file ***\n')