from pathlib import Path
import torch as trc
import sys

from do_regr import MLP2 as pointMLP
from plot import slices

model = pointMLP(4, 5)
model_dir = Path('/Users/jarl/2d-discharge-nn/created_models/mlp/2023-05-09_1134/M1b')
out_dir = Path('/Users/jarl/2d-discharge-nn')
model.load_state_dict(trc.load(model_dir))

slices(model, out_dir=out_dir)
