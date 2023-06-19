import torch.nn as nn
import torch.nn.functional as F
import torchvision

# 32 x 32
class MLP(nn.Module):
    """MLP to recreate encodings from a pair of V and P.
    """
    def __init__(self, input_size, output_size, dropout_prob) -> None:
        super(MLP, self).__init__()
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
        self.fc1 = nn.Linear(input_size, 512)
        # self.fc2 = nn.Linear(512, output_size)  # for output size of 2560, we halve the neurons per layer
        self.output = nn.Linear(512, output_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
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
    
