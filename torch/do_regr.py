
import torch

model = MLP()
model.load_state_dict(torch.load(filepath))
model.eval()
model(x) for x in dataframe