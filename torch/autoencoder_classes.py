import torch.nn as nn
import torch.nn.functional as F
import torchvision

class A212(nn.Module):
    """Autoencoder using square images as inputs.
    
    Input sizes are (5, 32, 32).
    """
    def __init__(self) -> None:
        super(A212, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(5, 10, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(10, 20, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(20, 20, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(20, 20, kernel_size=2, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(20, 10, kernel_size=2, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(10, 5, kernel_size=2, stride=2),
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        # decoded = torchvision.transforms.functional.crop(
        #     decoded, 0, 0, 64, 64)
        return decoded


class A300(nn.Module):
    """Autoencoder using square images as inputs.
    
    Input sizes are (5, 32, 32).
    """
    def __init__(self) -> None:
        super(A300, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(5, 10, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(10, 20, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(20, 20, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(20, 20, kernel_size=3, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(20, 10, kernel_size=5, stride=2),
            nn.ReLU(),

            nn.Conv2d(10, 10, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(10, 5, kernel_size=5, stride=2),
            nn.ReLU(),

            nn.Conv2d(5, 5, kernel_size=1, stride=1),
            nn.ReLU(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = torchvision.transforms.functional.crop(
             decoded, 0, 0, 32, 32)
        return decoded  

# model = A300()
# print(summary(model, input_size=(1, 5, 32, 32)))


class SquareAE64(nn.Module):
    """Autoencoder using square images as inputs.
    
    Input sizes are (5, 64, 64).
    """
    def __init__(self) -> None:
        super(SquareAE64, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(5, 10, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),

            nn.Conv2d(10, 20, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),

            nn.Conv2d(20, 40, kernel_size=1, stride=2, padding=0),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(40, 40, kernel_size=3, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(40, 20, kernel_size=5, stride=2),
            nn.ReLU(),

            nn.Conv2d(20, 20, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(20, 10, kernel_size=5, stride=2),
            nn.ReLU(),

            nn.Conv2d(10, 5, kernel_size=1, stride=1),
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = torchvision.transforms.functional.crop(
            decoded, 0, 0, 64, 64)
        return decoded
    

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(5, 10, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(20, 40, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(40, 20, kernel_size=3, stride=2),
            nn.ConvTranspose2d(20, 10, kernel_size=3, stride=2),
            nn.ConvTranspose2d(10, 5, kernel_size=5, stride=2)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = torchvision.transforms.functional.crop(
            decoded, 0, 0, 707, 200)
        return decoded