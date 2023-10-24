import torch
from torchinfo import summary

from pathlib import Path

from MLP import MLP

if __name__ == "__main__":
    name = 'test_MLP'
    model = MLP(name, 4, 5)
    model.eval()

    print(summary(model, input_size=(128, 4), dtypes=[torch.float64]))