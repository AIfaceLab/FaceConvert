import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as functional
from .Block_2 import Conv2D_2, SelfAttention, upscale_pixel, BasicBlock


tanH = torch.tanh
Sigmoid = torch.sigmoid
class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        # 3*256*256
        self.conv1 = Conv2D_2(3, 32, kernel_size=5, stride=2, padding=2)#32*128*128
        self.conv2 = Conv2D_2(32, 64, kernel_size=3, stride=2) # 64*64*64
        self.conv3 = Conv2D_2(64, 128, kernel_size=3, stride=2)#128*32*32
        self.conv4 = Conv2D_2(128, 256, kernel_size=3, stride=2)#256*16*16

        self.selfatten1 = SelfAttention(256)

        self.conv5 = Conv2D_2(256, 512, kernel_size=3, stride=2)#512*8*8

        self.selfatten2 = SelfAttention(512)

        self.conv6 = Conv2D_2(512, 1024, kernel_size=3, stride=2)#1024*4*4

        self.fc1 = nn.Linear(1024*4*4, 1024, bias=True)#1024
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 4*4*1024, bias=True)# 1024*4*4
        self.drop2 = nn.Dropout(0.5)

        self.upscale = upscale_pixel(1024, 512) # 1024*4*4 -> 512*8*8

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)

        out = self.selfatten1(out)
        out = self.conv4(out)
        out = self.selfatten2(out)
        out = self.conv5(out)

        out = out.view(x.size(0), -1)

        out = self.fc1(out)
        out = self.drop1(out)
        out = self.fc2(out)
        out = self.drop2(out)

        out = out.view(-1, 1024, 4, 4)

        out = self.upscale(out)

        return out



class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # input 512*64*64
        self.up1 = upscale_pixel(512, 256, 3, padding=1)# 256*128*16
        self.up2 = upscale_pixel(256, 128, 3, padding=1)# 128*32*32
        self.selfatten1 = SelfAttention(128)
        self.up3 = upscale_pixel(128, 64, 3, padding=1) # 64*64*64

        self.resblock = BasicBlock(64, 64, stride=1)

        self.selfatten2 = SelfAttention(64)
        self.inNorm1 = nn.InstanceNorm2d(3, momentum=0.1)
        self.inNorm2 = nn.InstanceNorm2d(1, momentum=0.1)
        self.conv1 = nn.Conv2d(64, 3, kernel_size=5, stride=1, padding=2)# BGR
        self.conv2 = nn.Conv2d(64, 1, kernel_size=5, stride=1, padding=2)# mask

    def forward(self, x):

        out = self.up1(x)
        out = self.up2(out)
        out = self.selfatten1(out)
        out = self.up3(out)

        out = self.resblock(out)

        out = self.selfatten2(out)

        BGR = self.conv1(out)
        BGR = self.inNorm1(BGR)
        BGR = 255*Sigmoid(BGR)

        mask = self.conv2(out)
        mask = self.inNorm2(mask)
        mask = tanH(mask)


        ABGR = torch.cat((BGR, mask), 1)

        return ABGR


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        #6*64*64
        self.conv1 = Conv2D_2(3, 64, kernel_size=3, stride=2)# 64*32*32
        self.conv2 = Conv2D_2(64, 128, kernel_size=3, stride=2)# 128*16*16

        self.selfatten1 = SelfAttention(128)

        self.conv3 = Conv2D_2(128, 256, kernel_size=3, stride=2)# 256*8*8

        self.selfatten2 = SelfAttention(256)

        self.conv4 = nn.Conv2d(256, 1, kernel_size=5, stride=1, padding=2)# 1*8*8

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        out = self.selfatten1(out)
        out = self.conv3(out)

        out = self.selfatten2(out)
        out = self.conv4(out)

        out = out.view(x.size(0), -1)

        return out














