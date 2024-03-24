import torch.nn as nn
import torch.nn.functional as F
import torchvision

from pathlib import Path

root = Path.cwd()
model_dir = root/'created_models'/'autoencoder'
data_dir = root/'data'/'interpolation_datasets'

class A212(nn.Module):
    """Autoencoder using square images as inputs.
    
    Input sizes are (5, 32, 32).
    """
    def __init__(self) -> None:
        super(A212, self).__init__()
        # trained on normalized dataset, otherwise see A212
        self.path = model_dir/'32x32'/'A212b'/'A212b'
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
        self.path = model_dir/'32x32'/'A300'/'A300'
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


class A300s(nn.Module):
    """Symmetrical variant of A300. Used to test if the additional conv2d layers make a difference.
    
    Input sizes are (5, 32, 32).
    """
    def __init__(self) -> None:
        super(A300s, self).__init__()
        self.path = model_dir/'32x32'/'A300s'/'A300s'
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

            nn.ConvTranspose2d(10, 5, kernel_size=5, stride=2),
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = torchvision.transforms.functional.crop(
             decoded, 0, 0, 32, 32)
        return decoded 
    

# 64x64 networks
class A64_7(nn.Module):
    """A64_6 with fixed padding of 1 and an extra convolutional layer at the decoder (idk why really)

    Size-matching is pending.
    
    Input sizes are (5, 64, 64).
    """
    def __init__(self) -> None:
        super(A64_7, self).__init__()
        self.path = model_dir/'64x64'/'A64-7'/'A64-7'
        self.encoder = nn.Sequential(
            nn.Conv2d(5, 10, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(10, 20, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(20, 40, kernel_size=1, stride=1, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(40, 40, kernel_size=3, stride=2),
            nn.ReLU(),

            nn.Conv2d(40, 40, kernel_size=(3, 3), stride=1, padding=1),
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


class A64_6(nn.Module):
    """Autoencoder using square images as inputs.
    
    Input sizes are (5, 64, 64).
    """
    def __init__(self) -> None:
        super(A64_6, self).__init__()
        self.path = model_dir/'64x64'/'A64-6'/'A64-6'
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

            # checkerboard patterns: https://distill.pub/2016/deconv-checkerboard/
            # 1. subpixel convolution: use a kernel size that is divisible by the stride to avoid 
            #       the overlap issue
            # 2. separate out the upsampling from the convolution to compute features
            #       for example, you might resize the image (using nearest neighbor ! or bilinear interpolation)
            #       and then do a convolutional layer (resize-convolution is implicityly weight-tying in 
            #       a way that discourages high frequency artifacts)
            #       TRY: torch.nn.Upsample('https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = torchvision.transforms.functional.crop(
            decoded, 0, 0, 64, 64)
        return decoded


class A64_6s(nn.Module):
    """Symmetric variant of A64_6.
    
    Input sizes are (5, 64, 64).
    """
    def __init__(self) -> None:
        super(A64_6s, self).__init__()
        self.path = model_dir/'64x64'/'A64-6s'/'A64-6s'
        self.resolution = 64
        self.encoder = nn.Sequential(
            nn.Conv2d(5, 10, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),

            nn.Conv2d(10, 20, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),

            nn.Conv2d(20, 40, kernel_size=1, stride=2, padding=0),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(40, 20, kernel_size=3, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(20, 10, kernel_size=5, stride=2),
            nn.ReLU(),

            # nn.Conv2d(20, 20, kernel_size=(3, 3), stride=1, padding=1),
            # nn.ReLU(),

            nn.ConvTranspose2d(10, 5, kernel_size=5, stride=2),
            nn.ReLU(),

            # nn.Conv2d(10, 5, kernel_size=1, stride=1),
            # nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = torchvision.transforms.functional.crop(
            decoded, 0, 0, 64, 64)
        return decoded


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
    

class A64_8(nn.Module):
    """A64_6 with larger input kernels. 
    I also separated the upsampling steps from the convolution steps.

    Reduction in sizes is achieved only with strided convolutions.
    by jarl @ 10 Oct 2023 18:00
    
    Input sizes are (5, 64, 64), encoded size is (40, 8, 8)
    """
    def __init__(self) -> None:
        super(A64_8, self).__init__()
        self.encoded_shape = (40, 8, 8)
        self.encoder = nn.Sequential(
            nn.Conv2d(5, 10, kernel_size=9, stride=2, padding=4),
            nn.ReLU(),

            nn.Conv2d(10, 20, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),

            nn.Conv2d(20, 40, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(40, 40, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(40, 20, kernel_size=5, padding=2, stride=1),
            nn.ReLU(),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(20, 10, kernel_size=(3, 3), padding=1, stride=1),
            nn.ReLU(),

            nn.Conv2d(10, 5, kernel_size=1, stride=1),
            nn.ReLU()

            # checkerboard patterns: https://distill.pub/2016/deconv-checkerboard/
            # 1. subpixel convolution: use a kernel size that is divisible by the stride to avoid 
            #       the overlap issue
            # 2. separate out the upsampling from the convolution to compute features
            #       for example, you might resize the image (using nearest neighbor ! or bilinear interpolation)
            #       and then do a convolutional layer (resize-convolution is implicityly weight-tying in 
            #       a way that discourages high frequency artifacts)
            #       TRY: torch.nn.Upsample('https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = torchvision.transforms.functional.crop(
            decoded, 0, 0, 64, 64)
        return decoded
    
class A64_9(nn.Module):
    """ *** best performing model ***
    A64_8 with all 3x3 kernels and MaxPool layers in the encoder.

    by jarl @ 25 Oct 2023 15:14
    
    This network takes longer to train due to 3x3 kernels and smaller strides.
    """
    def __init__(self) -> None:
        super(A64_9, self).__init__()
        self.path = model_dir/'64x64'/'A64-9'
        self.name = "A64-9"
        self.test_pair = (300, 60)
        self.val_pair = (400, 45)
        self.is_square = True
        self.in_resolution = 64
        self.ncfile = data_dir/'synthetic'/'synthetic_averaged999_s64.nc'
        self.encoder = nn.Sequential(
            nn.Conv2d(5, 10, kernel_size=3, stride=1, padding='same'),  # padding='same' maintains the output size
            nn.MaxPool2d(2, 2),
            nn.ReLU(),

            nn.Conv2d(10, 20, kernel_size=3, stride=1, padding='same'),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),

            nn.Conv2d(20, 40, kernel_size=3, stride=1, padding='same'),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(40, 40, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(40, 20, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(20, 10, kernel_size=(3, 3), padding=1, stride=1),
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
    

class A64_9_BN(nn.Module):
    """ A64_9 with BatchNorm.

    by jarl @ 15 Mar 2024 08:24
    """
    def __init__(self) -> None:
        super(A64_9_BN, self).__init__()

        #### train variables
        self.path = model_dir/'64x64'/'A64-9_BN'
        self.name = "A64-9-BN"
        self.test_pair = (300, 60)
        self.val_pair = (400, 45)
        self.is_square = True
        self.in_resolution = 64
        self.ncfile = data_dir/'synthetic'/'synthetic_averaged999_s64.nc'

        self.encoder = nn.Sequential(
            nn.Conv2d(5, 10, kernel_size=3, stride=1, padding='same'),  # padding='same' maintains the output size
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(10),
            nn.ReLU(),

            nn.Conv2d(10, 20, kernel_size=3, stride=1, padding='same'),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(20),
            nn.ReLU(),

            nn.Conv2d(20, 40, kernel_size=3, stride=1, padding='same'),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(40),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(40, 40, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(40),
            nn.ReLU(),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(40, 20, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(20),
            nn.ReLU(),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(20, 10, kernel_size=(3, 3), padding=1, stride=1),
            nn.BatchNorm2d(10),
            nn.ReLU(),

            nn.Conv2d(10, 5, kernel_size=1, stride=1),
            nn.BatchNorm2d(5),
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = torchvision.transforms.functional.crop(
            decoded, 0, 0, 64, 64)
        return decoded
    

class A200_1(nn.Module):
    """Autoencoder for 200x200 input images with all 3x3 kernels and MaxPool layers.

    by jarl @ 24 Jan 2023
    """
    def __init__(self) -> None:
        super(A200_1, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(5, 10, kernel_size=3, stride=1, padding='same'),  # padding='same' maintains the output size
            nn.MaxPool2d(2, 2),
            nn.ReLU(),

            nn.Conv2d(10, 20, kernel_size=3, stride=1, padding='same'),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),

            nn.Conv2d(20, 40, kernel_size=3, stride=1, padding='same'),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(40, 40, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(40, 20, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(20, 10, kernel_size=(3, 3), padding=1, stride=1),
            nn.ReLU(),

            nn.Conv2d(10, 5, kernel_size=1, stride=1),
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = torchvision.transforms.functional.crop(self.decoder(encoded), 0, 0, 200, 200)
        return decoded

class FullAE1(nn.Module):
    """Autoencoder for a whole 707x200 image.

    The objective is to reduce the number of data points 
    in the latent space to about ~2000.

    Args:
        nn (nn.Module): _description_
    """
    def __init__(self):
        super(FullAE1, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(5, 10, kernel_size=3, stride=1, padding='same'),  # padding='same' maintains the output size
            nn.MaxPool2d(2, 2),
            nn.ReLU(),

            nn.Conv2d(10, 20, kernel_size=3, stride=1, padding='same'),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),

            nn.Conv2d(20, 20, kernel_size=3, stride=1, padding='same'),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),

            nn.Conv2d(20, 20, kernel_size=3, stride=1, padding='same'),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),

            nn.Conv2d(20, 20, kernel_size=3, stride=1, padding='same'),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(20, 20, kernel_size=3, stride=1, padding='same'),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ReLU(),

            nn.Conv2d(20, 20, kernel_size=3, stride=1, padding='same'),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ReLU(),

            nn.Conv2d(20, 20, kernel_size=3, stride=1, padding='same'),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ReLU(),

            nn.Conv2d(20, 10, kernel_size=3, stride=1, padding='same'),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ReLU(),

            nn.Conv2d(10, 5, kernel_size=3, stride=1, padding='same'),
            nn.UpsamplingBilinear2d(size=(707, 200)),  # force output shape
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)  # encoded: (20, 22, 6)
        decoded = self.decoder(encoded)

        return decoded
