""" PointMLP classes """

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path 

root = Path.cwd()
model_dir = root/'created_models'/'pointmlp'

def get_model(name:str, input_size:int, output_size:int):
    # there has to be a better way of doing this
    model_string = f"{name}({input_size}, {output_size})"
    return eval(model_string)

class TunaA(nn.Module):
    def __init__(self, input_size:int, output_size:int) -> None:
        super(TunaA, self).__init__()
        self.path = model_dir/'TunaA'
        self.name = 'TunaA'
        self.input_size = input_size
        self.output_size = output_size

        self.dummy_param = nn.Parameter(torch.empty(0))

        # layers
        self.in_layer = nn.Linear(input_size, 115)
        self.hidden = [nn.Linear(115, 78), 
                  nn.Linear(78, 46),
                  nn.Linear(46, 26),
                  nn.Linear(26, 46),
                  nn.Linear(46, 82),
                  nn.Linear(82, 106),
                  nn.Linear(106, 115)]  
        self.fc2 = nn.Linear(115, 78)
        self.fc3 = nn.Linear(78, 46)
        self.fc4 = nn.Linear(46, 26)
        self.fc5 = nn.Linear(26, 46)
        self.fc6 = nn.Linear(46, 82)
        self.fc7 = nn.Linear(82, 106)
        self.fc8 = nn.Linear(106, 115)
        self.out_layer = nn.Linear(115, output_size)
        
    def forward(self, x):
        x = self.in_layer(x)
        x= F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)
        x = self.fc6(x)
        x = F.relu(x)
        x = self.fc7(x)
        x = F.relu(x)
        x = self.fc8(x)
        x = F.relu(x)
        
        # for layer in self.hidden:
        #     x = layer(x)
        #     x = F.relu(x)
        x = self.out_layer(x)
        output = F.relu(x)

        return output
