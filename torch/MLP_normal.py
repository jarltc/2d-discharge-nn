"""
Explore different data normalization for the MLP.

23 May 17:00
author @jarl
"""

import os
import sys
import time
import datetime
import shutil
import pickle
from tqdm import tqdm
from pathlib import Path
import torch.multiprocessing as mp

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import data_helpers as data
import plot

torch.set_default_dtype(torch.float64)

if __name__ == '__main__':
