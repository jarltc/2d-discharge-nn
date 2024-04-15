import torch
import numpy as np

import autoencoder_classes as AE
from torchinfo import summary
from data_helpers import set_device

if __name__ == "__main__":
    device = set_device()
    device_name = set_device(name=True)
    autoencoder = AE.get_model("A200_2")
    autoencoder.to(device)
    
    in_size=(1, 5, 200, 200)

    print(summary(autoencoder.encoder, input_size=in_size, device=device_name, verbose=2))