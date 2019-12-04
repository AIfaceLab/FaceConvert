import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as functional
from .Block import ResBlockDown, ResBlockUp, SelfAttention, BasicBlock

LeakyReLU = torch.leaky_relu
tanH = torch.tanh

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        # 3*256*256
        self.resDown1 = ResBlockDown(3, 64, conv_size=9, padding_size=4)  # out 64*128*128
        self.in1 = nn.InstanceNorm2d(64, affine=True)

        self.resDown2 = ResBlockDown(64, 128)  # out 128*64*64
        self.in2 = nn.InstanceNorm2d(128, affine=True)

        self.resDown3 = ResBlockDown(128, 256)  # out 256*32*32
        self.in3 = nn.InstanceNorm2d(256, affine=True)

        self.self_att_Down = SelfAttention(256)  # out 256*32*32

        self.resDown4 = ResBlockDown(256, 512)  # out 512*16*16
        self.in4 = nn.InstanceNorm2d(512, affine=True)

        self.resDown5 = ResBlockDown(512, 512)  # out 512*8*8
        self.in5 = nn.InstanceNorm2d(512, affine=True)

        self.resDown6 = ResBlockDown(512, 512)  # out 512*4*4
        self.in6 = nn.InstanceNorm2d(512, affine=True)

        self.resblock1 = BasicBlock(4, 4, stride=1)
        self.resblock2 = BasicBlock(4, 4, stride=1)

    def forward(self, x):
        out = self.resDown1(x)
        out = self.in1(out)

        out = self.resDown2(out)
        out = self.in2(out)

        out = self.resDown3(out)
        out = self.in3(out)

        out = self.self_att_Down(out)

        out = self.resDown4(out)
        out = self.in4(out)

        out = self.resDown5(out)
        out = self.in5(out)

        out = self.resDown6(out)
        out = self.in6(out)

        out = self.resblock1(out)
        out = self.resblock2(out)

        return out


class Decoder(nn.Module):
    in_height = 256

    def __init__(self):
        # 512*4*4
        super(Decoder, self).__init__()
        self.resUp1 = ResBlockUp(512, 512)  # out 512*8*8
        self.resUp2 = ResBlockUp(512, 512)  # out 512*16*16
        self.resUp3 = ResBlockUp(512, 256)  # out 256*32*32
        self.resUp4 = ResBlockUp(256, 128)  # out 128*64*64

        self.self_att_Up = SelfAttention(128)  # out 128*64*64

        self.resUp5 = ResBlockUp(128, 64)  # out 64*128*128
        self.resUp6 = ResBlockUp(64, 3, out_size=(self.in_height, self.in_height), scale=None, conv_size=9, padding_size=4) #out 3*224*224

    def forward(self, x):
        out = self.resUp1(x)

        out = self.resUp2(out)

        out = self.resUp3(out)

        out = self.resUp4(out)

        out = self.self_att_Up(out)

        out = self.resUp5(out)

        out = self.resUp6(out)
        out = self.sigmoid(out)

        out = out * 255

        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.relu = nn.ReLU(inplace=False)

        # in 6*224*224
        self.resDown1 = ResBlockDown(6, 64)  # out 64*128*128
        self.resDown2 = ResBlockDown(64, 128)  # out 128*64*64
        self.resDown3 = ResBlockDown(128, 256)  # out 256*32*32
        self.self_att = SelfAttention(256)  # out 256*32*32
        self.resDown4 = ResBlockDown(256, 512)  # out 512*16*16
        self.resDown5 = ResBlockDown(512, 512)  # out 512*8*8
        self.resDown6 = ResBlockDown(512, 512)  # out 512*4*4
        self.res = BasicBlock(512, 512)  # out 512*4*4
        self.sum_pooling = nn.AdaptiveAvgPool2d((1, 1))  # out 512*1*1

        self.W_i = nn.Parameter(torch.rand(512, 1))
        self.w_0 = nn.Parameter(torch.randn(512, 1))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self,x):

        out1 = self.resDown1(x)

        out2 = self.resDown2(out1)

        out3 = self.resDown3(out2)

        out = self.self_att(out3)

        out4 = self.resDown4(out)

        out5 = self.resDown5(out4)

        out6 = self.resDown6(out5)

        out7 = self.res(out6)

        out = self.sum_pooling(out7)

        out = out.view(-1, 512, 1)  # out B*512*1

        out = torch.bmm(out.transpose(1, 2), (self.W_i[:, i].unsqueeze(-1)).transpose(0, 1) + self.w_0) + self.b

        # 1x1