from pathlib import Path
import torch as trc
import sys

from do_regr import MLP, MLP2
from plot import slices

try:
    pointMLP = MLP
    model = pointMLP(4, 5)
except:
    pointMLP = MLP2
    model = pointMLP(4, 5)
    
model_dir = Path(input('Enter path to model dir: '))
weights = next(model_dir.rglob('M*'))  # get first item in generator
model.load_state_dict(trc.load(weights))

slices(model, model_dir/'scalers', out_dir=model_dir)
 