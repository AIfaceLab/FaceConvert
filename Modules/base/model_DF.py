import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as functional
from .Block_DF import Upscale, DownScale


class Encoder(nn.Module):
    def __init__(self, shape):
        super(Encoder, self).__init__()
        #3*256*256 //  3*128*128
        self.shape = shape
        if self.shape == 256:
            self.downscale = DownScale(3, 64)  # 64*128*128
            self.downscale1 = DownScale(64, 128)  # 128*64*64
        elif self.shape == 128:
            self.downscale1 = DownScale(3, 128)  # 128*64*64
        else:
            print("Don't support the image size!! :=> ", shape)
        self.downscale2 = DownScale(128, 256)  # 256*32*32
        self.downscale3 = DownScale(256, 512)  # 512*16*16
        self.downscale4 = DownScale(512, 1024)  # 1024*8*8

        self.dense1 = nn.Linear(1024*8*8, 512)
        self.drop1 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(512, 8*8*512)  # 512*8*8
        self.drop2 = nn.Dropout(0.5)

        self.upscale = Upscale(512, 512)  # 512*16*16

    def forward(self, x):
        if self.shape == 256:
            out = self.downscale(x)
            out = self.downscale1(out)
        elif self.shape == 128:
            out = self.downscale1(x)

        out = self.downscale2(out)
        out = self.downscale3(out)
        out = self.downscale4(out)

        out = out.view(x.size(0), -1)
        out = self.dense1(out)
        out = self.drop1(out)
        out = self.dense2(out)
        out = self.drop2(out)
        out = out.view(-1, 512, 8, 8)

        out = self.upscale(out)

        return out


class Decoder(nn.Module):
    def __init__(self, shape):
        super(Decoder, self).__init__()
        self.shape = shape
        self.Sigmoid = torch.sigmoid
        self.upscale1 = Upscale(512, 512)  # 512*32*32
        self.upscale2 = Upscale(512, 256)  # 256*64*64
        self.upscale3 = Upscale(256, 128)  # 128*128*128
        if self.shape == 256:
            self.upscale4 = Upscale(128, 64)  # 64*256*256
            self.conv = nn.Conv2d(64, 3, kernel_size=5,
                                  stride=1, padding=2)  # 3*256*256
        elif self.shape == 128:
            self.conv = nn.Conv2d(128, 3, kernel_size=5,
                                  stride=1, padding=2)  # 1*128*128
        else:
            print("Don't support the image size!! :=> ", shape)

        self.upscale1_m = Upscale(512, 512)  # 512*32*32
        self.upscale2_m = Upscale(512, 256)  # 256*64*64
        self.upscale3_m = Upscale(256, 128)  # 128*128*128
        if self.shape == 256:
            self.upscale4_m = Upscale(128, 64)  # 64*256*256
            self.conv_m = nn.Conv2d(
                64, 1, kernel_size=5, stride=1, padding=2)  # 1*256*256
        elif self.shape == 128:
            self.conv_m = nn.Conv2d(
                128, 1, kernel_size=5, stride=1, padding=2)  # 1*128*128

    def forward(self, x):
        out1 = self.upscale1(x)
        out1 = self.upscale2(out1)
        out1 = self.upscale3(out1)
        if self.shape == 256:
            out1 = self.upscale4(out1)
            out1 = self.conv(out1)
            out1 = self.Sigmoid(out1)
        elif self.shape == 128:
            out1 = self.conv(out1)
            out1 = self.Sigmoid(out1)
        BGR = 255*out1

        out2 = self.upscale1_m(x)
        out2 = self.upscale2_m(out2)
        out2 = self.upscale3_m(out2)
        if self.shape == 256:
            out2 = self.upscale4_m(out2)
            out2 = self.conv_m(out2)
            out2 = self.Sigmoid(out2)
        elif self.shape == 128:
            out2 = self.conv_m(out2)
            out2 = self.Sigmoid(out2)
        mask = out2

        ABGR = torch.cat((BGR, mask), 1)
        return ABGR
