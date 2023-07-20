from pathlib import Path
import torch as trc
import sys

from do_regr import MLP2 as pointMLP
from plot import slices

model = pointMLP(4, 5)
model_dir = Path(input('Enter path to model dir: '))
model.load_state_dict(trc.load(model_dir/'M1b'))

slices(model, model_dir/'scalers', out_dir=model_dir)
 