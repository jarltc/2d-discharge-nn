# Plotting functions module

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

def save_history_graph(history, out_dir, param='mae'):
    # TODO: move to plot module
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