import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as functional
from .Block import Conv2D, DownSample_1, BasicBlock, Dconv2D

LeakyReLU = functional.leaky_relu
tanH = functional.tanh
class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        # input: 3*256*256
        self.conv1 = Conv2D(3, 64, kernel_size=7, stride=1, padding=3) #out: 64*256*256
        self.conv2 = Conv2D(64, 128, kernel_size=3, stride=2)#out: 128*128*128
        self.conv3 = Conv2D(128, 256, kernel_size=3, stride=2)#out: 256* 64 *64

        self.downsample = DownSample_1(256, 512)
        self.resblk1 = BasicBlock(256, 512, stride=1, downsample=self.downsample)
        self.resblk2 = BasicBlock(512, 512, stride=1)
        self.resblk3 = BasicBlock(512, 512, stride=1)
        self.resblk4 = BasicBlock(512, 512, stride=1)

    def get_shared_layer(self):
        return self.resblk4

    def forward(self, x):

        out1 = self.conv1(x)
        out1 = LeakyReLU(out1)

        out2 = self.conv2(out1)
        out2 = LeakyReLU(out2)

        out3 = self.conv3(out2)
        out3 = LeakyReLU(out3)

        out = self.resblk1(out3)
        out = self.resblk2(out)
        out = self.resblk3(out)
        out = self.resblk4(out)


class Generator(nn.Module):
    def __init__(self, shared_layer):
        super(Generator, self).__init__()

        #input: 512*64*64
        self.resblk1 = shared_layer
        self.resblk2 = BasicBlock(512, 512, stride=1)
        self.resblk3 = BasicBlock(512, 512, stride=1)
        self.resblk4 = BasicBlock(512, 512, stride=1)

        self.Dconv1 = Dconv2D(512, 256, kernel_size=3, stride=2)# out: 256*128*128
        self.Dconv2 = Dconv2D(256, 128, kernel_size=3, stride=2)# out:128*256*256
        self.Dconv3 = Dconv2D(128,3, kernel_size=3,stride=1)# out: 3*256*256

    def forward(self, x):
        out = self.resblk1(x)
        out = self.resblk2(out)
        out = self.resblk3(out)
        out = self.resblk4(out)

        out1 = self.Dconv1(out)
        out1 = LeakyReLU(out1)

        out2 = self.Dconv2(out1)
        out2 = LeakyReLU(out2)

        out3 = self.Dconv3(out2)
        out3 = tanH(out3)
        return out3


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        #input: 3*256*256
        self.Sigmoid = nn.Sigmoid()
        self.Conv1 = Conv2D(3, 64, kernel_size=3, stride=2)
        self.Conv2 = Conv2D(64,128, kernel_size=3, stride=2)
        self.Conv3 = Conv2D(128, 256, kernel_size=3, stride=2)
        self.Conv2 = Conv2D(256, 512, kernel_size=3, stride=2)
        self.Conv2 = Conv2D(512, 1024, kernel_size=3, stride=2)
        self.Conv2 = Conv2D(1024, 1, kernel_size=2, stride=1)

    def forward(self, x):

        out1 = self.Conv1(x)
        out1 = LeakyReLU(out1)

        out2 = self.Conv1(out1)
        out2 = LeakyReLU(out2)

        out3 = self.Conv1(out2)
        out3 = LeakyReLU(out3)

        out4 = self.Conv1(out3)
        out4 = LeakyReLU(out4)

        out5 = self.Conv1(out4)
        out5 = LeakyReLU(out5)

        out6 = self.Conv1(out5)
        out6 = self.Sigmoid(out6)

        return out6



