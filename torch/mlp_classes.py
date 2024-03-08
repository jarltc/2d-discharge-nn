import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pathlib import Path

root = Path.cwd()
model_dir = root/'created_models'/'conditional_autoencoder'

# 32 x 32
class MLP(nn.Module):
    """MLP to recreate encodings from a pair of V and P.
    """
    def __init__(self, input_size, output_size, dropout_prob) -> None:
        super(MLP, self).__init__()
        self.path64 = model_dir/'64x64'/'A64g'/'A64g'
        self.path32 = model_dir/'32x32'/'A32g'/'A32g'
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

    dont use this yet lol
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
    

class MLP2(nn.Module):
    """MLP with no dropout
    """
    def __init__(self, input_size, output_size, dropout_prob) -> None:
        super(MLP2, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, 40)
        self.fc2 = nn.Linear(40, 80)
        self.fc3 = nn.Linear(80, 160)
        self.fc4 = nn.Linear(160, 320)
        self.fc5 = nn.Linear(320, 640)
        self.fc6 = nn.Linear(640, 1280)
        self.fc7 = nn.Linear(1280, output_size)  # for output size of 2560, we halve the neurons per layer

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
        output = F.relu(x)

        return output
    

class MLP3(nn.Module):
    """MLP1 with more nodes?
    """
    def __init__(self, input_size, output_size, dropout_prob) -> None:
        super(MLP3, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.input = nn.Linear(input_size, 1024)
        self.fc1 = nn.Linear(1024, 1024)
        
        self.output = nn.Linear(1024, output_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = F.relu(self.input(x))

        for _ in range(6):
            x = self.fc1(x)
            x = F.relu(x)

        output = F.relu(self.output(x))

        return output
    

class MLP4(nn.Module):
    """MLP3 with LESS nodes?
    """
    def __init__(self, input_size, output_size, dropout_prob) -> None:
        super(MLP4, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.input = nn.Linear(input_size, 20)

        self.fc1 = nn.Linear(20, 40)
        self.fc2 = nn.Linear(40, 80)
        self.fc3 = nn.Linear(80, 100)
        self.fc4 = nn.Linear(100, 120)
        
        self.output = nn.Linear(120, output_size)

    def forward(self, x):
        x = F.relu(self.input(x))

        # hidden layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        output = F.relu(self.output(x))

        return output