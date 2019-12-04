import torch
import torch.nn as nn



class DownScale(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, strides=2, padding=2):
        super(DownScale, self).__init__()
        self.LeakyReLU = nn.LeakyReLU(0.1)
        self.Conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=strides,
                              padding=padding)
        # self.normal = nn.InstanceNorm2d(out_channels)
        self.normal = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.Conv(x)
        out = self.LeakyReLU(out)
        out = self.normal(out)
        return out

class Upscale(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, strides=1, padding=1):
        super(Upscale, self).__init__()
        self.LeakyReLU = nn.LeakyReLU(0.1)
        self.Conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels*4,
                              kernel_size=kernel_size,
                              stride=strides,
                              padding=padding)
        self.PixelShuffler = nn.PixelShuffle(2)
        # self.normal = nn.InstanceNorm2d(out_channels)
        self.normal = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.Conv(x)
        out = self.LeakyReLU(out)
        out = self.PixelShuffler(out)
        out = self.normal(out)

        return out