import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, shape=128):
        super(Encoder, self).__init__()
        self.shape = shape
        self.downscale_128 = nn.Sequential(
            nn.Conv2d(3, 128, 5, stride=2, padding=2),
            nn.LeakyReLU(0.1)  # in_place = True or False
        )
        self.downscale_256 = nn.Sequential(
            nn.Conv2d(128, 256, 5, stride=2, padding=2),
            nn.LeakyReLU(0.1)  # in_place = True or False
        )
        self.downscale_512 = nn.Sequential(
            nn.Conv2d(256, 512, 5, stride=2, padding=2),
            nn.LeakyReLU(0.1)  # in_place = True or False
        )
        self.downscale_1024 = nn.Sequential(
            nn.Conv2d(512, 1024, 5, stride=2, padding=2),
            nn.LeakyReLU(0.1)  # in_place = True or False
        )
        if self.shape == 256:
            self.dense_512 = nn.Linear(1024*16*16, 512)
        else:
            self.dense_512 = nn.Linear(1024*8*8, 512)
        self.dense_8_8_512 = nn.Linear(512, 8*8*512)
        self.upscale = nn.Sequential(
            nn.Conv2d(512, 512*4, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        x = self.downscale_128(x)
        x = self.downscale_256(x)
        x = self.downscale_512(x)
        x = self.downscale_1024(x)
        x = x.view(x.size(0), -1)
        x = self.dense_512(x)
        x = self.dense_8_8_512(x)
        x = x.view(x.size(0), 512, 8, 8)
        x = self.upscale(x)
        return x


class Decoder(nn.Module):
    def __init__(self, shape=128):
        super(Decoder, self).__init__()
        self.shape = shape
        # self.upscale_512
        self.upscale_512 = nn.Sequential(
            nn.Conv2d(512, 512*4, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(512)
        )

        self.upscale_256 = nn.Sequential(
            nn.Conv2d(512, 256*4, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(256)
        )
        self.upscale_128 = nn.Sequential(
            nn.Conv2d(256, 128*4, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(128)
        )
        if self.shape == 256:
            self.upscale_64 = nn.Sequential(
                nn.Conv2d(128, 64*4, 3, stride=1, padding=1),
                nn.LeakyReLU(0.1),
                nn.PixelShuffle(2),
                nn.BatchNorm2d(64)
            )
            self.conv2d = nn.Conv2d(64, 3, 5, padding=2)
        else:
            self.conv2d = nn.Conv2d(128, 3, 5, padding=2)
        self.out_layer = nn.Sigmoid()
        # --------------------------------------------
        self.upscale_512_mask = nn.Sequential(
            nn.Conv2d(512, 512*4, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(512)
        )

        self.upscale_256_mask = nn.Sequential(
            nn.Conv2d(512, 256*4, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(256)
        )
        self.upscale_128_mask = nn.Sequential(
            nn.Conv2d(256, 128*4, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(128)
        )
        if self.shape == 256:
            self.upscale_64_mask = nn.Sequential(
                nn.Conv2d(128, 64*4, 3, stride=1, padding=1),
                nn.LeakyReLU(0.1),
                nn.PixelShuffle(2),
                nn.BatchNorm2d(64)
            )
            self.conv2d_mask = nn.Conv2d(64, 1, 5, padding=2)
        else:
            self.conv2d_mask = nn.Conv2d(128, 1, 5, padding=2)

        # self.conv2d_mask = nn.Conv2d(128, 1, 5, padding=2)
        self.out_layer_mask = nn.Sigmoid()

    def forward(self, x):
        out = self.upscale_512(x)
        out = self.upscale_256(out)
        out = self.upscale_128(out)
        if self.shape == 256:
            out = self.upscale_64(out)
        out = self.conv2d(out)
        out = self.out_layer(out)

        out_mask = self.upscale_512_mask(x)
        out_mask = self.upscale_256_mask(out_mask)
        out_mask = self.upscale_128_mask(out_mask)
        if self.shape == 256:
            out_mask = self.upscale_64_mask(out_mask)
        out_mask = self.conv2d_mask(out_mask)
        out_mask = self.out_layer_mask(out_mask)
        return out, out_mask
