"""
Python code to update my self-made database
"""

import sqlite3
from pathlib import Path
from datetime import datetime
import re

def get_metadata(model_dir:Path):
    file = model_dir/'train_log.txt'
    epochs = None
    resolution = None

    with open(file, 'r') as file:
        for line in file:
            match1 = re.search(r'Epochs: ', line)
            match2 = re.search(r'Resolution: ', line)
            if match1:
                epochs = line.split()[1]
            elif match2:
                resolution = line.split()[1]

    return epochs, resolution


def get_root_dirs(folder:Path):
    """Get the folders that have no subdirectories (meaning they only contain a model)

    Args:
        folder (Path): Folder to search through.

    Returns:
        root_dirs: List of Paths to the root folders
    """
    root_dirs = []
    for subdir in folder.rglob(''):
        # any() checks if any of the items returned by iterdir() are directories,
        # returns True if there are any subdirs.
        if subdir.is_dir() & ~any(subsubdir.is_dir() for subsubdir in subdir.iterdir()):
            root_dirs.append(subdir)

    return root_dirs


def get_iteration(folder:Path):
    """Get the iteration number for a model

    Args:
        folder (Path): _description_
    """


if __name__ == "__main__":

    root  = Path('/Users/jarl/2d-discharge-nn/created_models')
    conn = sqlite3.connect(root/'created_models.db')
    cursor = conn.cursor()

    autoencoders = root/'autoencoder'
    patternA = r"A\d{3}"  # Axxx where x is a digit from 0-9
    patternB = r"\bA\d{2}\b"  # Axxx where x is a digit from 0-9
    patternC = r"A\d{3}[a-z]"  # A212b
    patternD = r"A\d{2}-\d{1}"  # A64-1

    root_dirs = get_root_dirs(autoencoders)
    for folder in root_dirs:
        if re.search(patternC, folder.name):
            base = folder.name[:-1]
            iteration = ord(folder.name[-1]) - 96  # 'a' is the 97th character 
        elif re.search(patternA, folder.name):
            base = folder.name
            iteration = None
        elif re.search(patternD, folder.name):
            base = folder.name.split('-')[0]
            iteration = str(eval(folder.name[-1]) - 1)
        elif re.search(patternB, folder.name):
            base = folder.name
            iteration = None
        else:
            base = folder.name
            iteration = None

        if (folder/'train_log.txt').exists():
            epochs, resolution = get_metadata(folder)
        else: 
            epochs = None
            resolution = None

        creation_time = root_dirs[0].stat().st_ctime
        formatted_time = datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d %H:%M')

        # (model name, path, type, iteration, base, creation_date, epochs, v_excl, p_excl, resolution)
        values = (folder.name, str(folder), 'autoencoder', iteration, base, formatted_time, epochs, '300', '60', resolution)
        cursor.execute('INSERT OR IGNORE INTO ae_models (name, path, type, iteration, base, \
                       creation_date, train_epochs, v_excluded, p_excluded, resolution)\
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                    values)

    # conn.commit()
    conn.close()
