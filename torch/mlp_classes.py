import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pathlib import Path

# 32 x 32
class MLP(nn.Module):
    """MLP to recreate encodings from a pair of V and P.
    """
    def __init__(self, input_size, output_size, dropout_prob) -> None:
        super(MLP, self).__init__()
        self.path64 = Path('/Users/jarl/2d-discharge-nn/created_models/conditional_autoencoder/64x64/A64g/A64g')
        self.path32 = Path('/Users/jarl/2d-discharge-nn/created_models/conditional_autoencoder/32x32/A32g/A32g')
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, output_size)  # for output size of 2560, we halve the neurons per layer
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc4(x)
        output = F.relu(x)

        return output


class MLP1(nn.Module):
    """MLP to recreate encodings from a pair of V and P.
    """
    def __init__(self, input_size, output_size, dropout_prob) -> None:
        super(MLP1, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.input = nn.Linear(input_size, 512)
        self.fc1 = nn.Linear(512, 512)
        
        self.output = nn.Linear(512, output_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = F.relu(self.input(x))

        for _ in range(5):
            x = self.fc1(x)
            x = F.relu(x)

        output = F.relu(self.output(x))

        return output


# 64 x 64
class MLP64(nn.Module):
    """MLP to recreate encodings from a pair of V and P.
    """
    def __init__(self, input_size, output_size, dropout_prob) -> None:
        super(MLP64, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, 80)
        self.fc2 = nn.Linear(80, 160)
        self.fc3 = nn.Linear(160, 320)
        self.fc4 = nn.Linear(320, 640)
        self.fc5 = nn.Linear(640, 1280)
        self.fc6 = nn.Linear(1280, output_size)  # for output size of 2560, we halve the neurons per layer
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc4(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc5(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc6(x)
        output = F.relu(x)

        return output
    
class pointMLP(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super(pointMLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, 115)  # linear: y = Ax + b
        self.fc2 = nn.Linear(115, 78)
        self.fc3 = nn.Linear(78, 26)
        self.fc4 = nn.Linear(26, 46)
        self.fc5 = nn.Linear(46, 82)
        self.fc6 = nn.Linear(82, 106)
        self.fc7 = nn.Linear(106, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
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
        
        output = x = F.relu(x)
        return output


# class pointMLP(nn.Module):
#     """Neural network model for point-wise prediction of 2D profiles.

#     Model architecture optimized using OpTuna. (7 hidden layers)
#     Args:
#         name (string): Model name
#         input_size (int): Size of input vector. Should be 4 for (V, P, x, y).
#         output_size (int): Size of output vector. Should be 5 for the usual variables.
#     """
#     def __init__(self, name, input_size, output_size) -> None:
#         super(pointMLP, self).__init__()
#         self.name = name
#         self.input_size = input_size
#         self.output_size = output_size

#         # specify number of hidden layers and the number of nodes in each
#         self.nodes = [115, 78, 26, 46, 82, 106]
#         num_hl = len(self.nodes)
#         self.in_layer = nn.Linear(self.input_size, self.nodes[0])  # linear: y = Ax + b
#         self.out_layer = nn.Linear(self.nodes[-1], self.output_size)

#         # generate list of hidden layers
#         self.hidden_layers = [nn.Linear(self.nodes[i], self.nodes[i+1]) for i in range(num_hl-1)]
        
#     def forward(self, x):
#         """Execute the forward pass.

#         Args:
#             x (torch.Tensor): Input tensor of size (batch_size, input_size)

#         Returns:
#             torch.Tensor: Predicted values given an input x.
#         """
#         x = self.in_layer(x)
#         x = F.relu(x)
        
#         # hidden layers
#         for layer in self.hidden_layers:
#             x = layer(x)
#             x = F.relu(x)

#         x = self.out_layer(x)
#         output = F.relu(x)
#         return output
    

class pointMLP2(nn.Module):
    """Neural network model for point-wise prediction of 2D profiles.

    Model architecture optimized using OpTuna. (7 hidden layers)
    Args:
        name (string): Model name
        input_size (int): Size of input vector. Should be 4 for (V, P, x, y).
        output_size (int): Size of output vector. Should be 5 for the usual variables.
    """
    def __init__(self, name, input_size, output_size) -> None:
        super(pointMLP2, self).__init__()
        self.name = name
        self.input_size = input_size
        self.output_size = output_size

        # specify number of hidden layers and the number of nodes in each
        self.nodes = [115, 78, 46, 26, 46, 82, 106, 115]
        num_hl = len(self.nodes)
        self.in_layer = nn.Linear(self.input_size, self.nodes[0])  # linear: y = Ax + b
        self.out_layer = nn.Linear(self.nodes[-1], self.output_size)

        # generate list of hidden layers
        self.hidden_layers = [nn.Linear(self.nodes[i], self.nodes[i+1]) for i in range(num_hl-1)]
        
    def forward(self, x):
        """Execute the forward pass.

        Args:
            x (torch.Tensor): Input tensor of size (batch_size, input_size)

        Returns:
            torch.Tensor: Predicted values given an input x.
        """
        x = self.in_layer(x)
        x = F.relu(x)
        
        # hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
            x = F.relu(x)

        x = self.out_layer(x)
        output = F.relu(x)
        return output
    