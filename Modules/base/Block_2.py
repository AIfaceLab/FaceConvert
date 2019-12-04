import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


BN_MOMENTUM = 0.01
IN_MOMENTUM = 0.1

LeakyReLU = F.leaky_relu


class Conv2D_2(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super(Conv2D_2, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channel,
                              out_channels=out_channel,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding
                              )
        self.LeakyReLU = nn.LeakyReLU(0.1)
        self.inNorm = nn.InstanceNorm2d(out_channel, eps=1e-05, momentum= IN_MOMENTUM)


    def forward(self, input_layer):
        x = input_layer

        out = self.conv(x)
        out = self.LeakyReLU(out)
        out = self.inNorm(out)

        return out

class SelfAttention(nn.Module):
    def __init__(self, in_channel):
        super(SelfAttention, self).__init__()

        # conv f
        self.conv_f = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel // 8, 1))
        # conv_g
        self.conv_g = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel // 8, 1))
        # conv_h
        self.conv_h = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 1))

        self.softmax = nn.Softmax(-2)  # sum in column j = 1
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        f_projection = self.conv_f(x)  # BxC'xHxW, C'=C//8
        g_projection = self.conv_g(x)  # BxC'xHxW
        h_projection = self.conv_h(x)  # BxCxHxW

        f_projection = torch.transpose(f_projection.view(B, -1, H * W), 1, 2)  # BxNxC', N=H*W
        g_projection = g_projection.view(B, -1, H * W)  # BxC'xN
        h_projection = h_projection.view(B, -1, H * W)  # BxCxN

        attention_map = torch.bmm(f_projection, g_projection)  # BxNxN
        attention_map = self.softmax(attention_map)  # sum_i_N (A i,j) = 1

        # sum_i_N (A i,j) = 1 hence oj = (HxAj) is a weighted sum of input columns
        out = torch.bmm(h_projection, attention_map)  # BxCxN
        out = out.view(B, C, H, W)

        out = self.gamma * out + x
        return out


class upscale_pixel(nn.Module):
    def __init__(self, inchannel, outchannel, kernel=3, padding=1):
        super(upscale_pixel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inchannel,
                               out_channels=outchannel*4,
                               kernel_size=kernel,
                               stride=1,
                               padding=padding)
        self.inNorm = nn.InstanceNorm2d(4*outchannel, momentum=IN_MOMENTUM)
        self.subpixel = nn.PixelShuffle(2)

    def forward(self, x):

        out = self.conv1(x)
        out = LeakyReLU(out)
        out = self.inNorm(out)
        out = self.subpixel(out)


        return out

class upscale_dconv(nn.Module):
    def __init__(self,  inchannel, outchannel, kernel=3, stride=2, padding=1):
        super(upscale_dconv, self).__init__()

        self.deconv = nn.ConvTranspose2d(in_channels=inchannel,
                                         out_channels=outchannel,
                                         kernel_size=kernel,
                                         stride=stride,
                                         padding=padding)
        self.inNorm = nn.InstanceNorm2d(outchannel, momentum=IN_MOMENTUM)
    def forward(self):
        pass



class Conv2D3(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super(Conv2D3, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channel,
                              out_channels=out_channel,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding
                              )

    def forward(self, input_layer):
        x = input_layer

        out = self.conv(x)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = Conv2D3(inplanes, planes, stride=stride)
        self.in1 = nn.InstanceNorm2d(planes, momentum=BN_MOMENTUM)
        #self.relu = nn.ReLU(inplace=False)

        self.conv2 = Conv2D3(planes, planes, stride=stride)
        self.in2 = nn.InstanceNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.in1(x)
        x = LeakyReLU(x)

        x = self.conv2(x)
        x = self.in2(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = x + residual
        out = LeakyReLU(out)

        return out


class Cropped_VGG19(nn.Module):
    def __init__(self):
        super(Cropped_VGG19, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 64, 3)
        self.conv1_2 = nn.Conv2d(64, 64, 3)
        self.conv2_1 = nn.Conv2d(64, 128, 3)
        self.conv2_2 = nn.Conv2d(128, 128, 3)
        self.conv3_1 = nn.Conv2d(128, 256, 3)
        self.conv3_2 = nn.Conv2d(256, 256, 3)
        self.conv3_3 = nn.Conv2d(256, 256, 3)
        self.conv4_1 = nn.Conv2d(256, 512, 3)
        self.conv4_2 = nn.Conv2d(512, 512, 3)
        self.conv4_3 = nn.Conv2d(512, 512, 3)
        self.conv5_1 = nn.Conv2d(512, 512, 3)
        # self.conv5_2 = nn.Conv2d(512,512,3)
        # self.conv5_3 = nn.Conv2d(512,512,3)

    def forward(self, x):
        conv1_1_pad = F.pad(x, (1, 1, 1, 1))
        conv1_1 = self.conv1_1(conv1_1_pad)
        relu1_1 = F.relu(conv1_1)
        conv1_2_pad = F.pad(relu1_1, (1, 1, 1, 1))
        conv1_2 = self.conv1_2(conv1_2_pad)
        relu1_2 = F.relu(conv1_2)
        pool1_pad = F.pad(relu1_2, (0, 1, 0, 1), value=float('-inf'))
        pool1 = F.max_pool2d(pool1_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        conv2_1_pad = F.pad(pool1, (1, 1, 1, 1))
        conv2_1 = self.conv2_1(conv2_1_pad)
        relu2_1 = F.relu(conv2_1)
        conv2_2_pad = F.pad(relu2_1, (1, 1, 1, 1))
        conv2_2 = self.conv2_2(conv2_2_pad)
        relu2_2 = F.relu(conv2_2)
        pool2_pad = F.pad(relu2_2, (0, 1, 0, 1), value=float('-inf'))
        pool2 = F.max_pool2d(pool2_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        conv3_1_pad = F.pad(pool2, (1, 1, 1, 1))
        conv3_1 = self.conv3_1(conv3_1_pad)
        relu3_1 = F.relu(conv3_1)
        conv3_2_pad = F.pad(relu3_1, (1, 1, 1, 1))
        conv3_2 = self.conv3_2(conv3_2_pad)
        relu3_2 = F.relu(conv3_2)
        conv3_3_pad = F.pad(relu3_2, (1, 1, 1, 1))
        conv3_3 = self.conv3_3(conv3_3_pad)
        relu3_3 = F.relu(conv3_3)
        pool3_pad = F.pad(relu3_3, (0, 1, 0, 1), value=float('-inf'))
        pool3 = F.max_pool2d(pool3_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        conv4_1_pad = F.pad(pool3, (1, 1, 1, 1))
        conv4_1 = self.conv4_1(conv4_1_pad)
        relu4_1 = F.relu(conv4_1)
        conv4_2_pad = F.pad(relu4_1, (1, 1, 1, 1))
        conv4_2 = self.conv4_2(conv4_2_pad)
        relu4_2 = F.relu(conv4_2)
        conv4_3_pad = F.pad(relu4_2, (1, 1, 1, 1))
        conv4_3 = self.conv4_3(conv4_3_pad)
        relu4_3 = F.relu(conv4_3)
        pool4_pad = F.pad(relu4_3, (0, 1, 0, 1), value=float('-inf'))
        pool4 = F.max_pool2d(pool4_pad, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        conv5_1_pad = F.pad(pool4, (1, 1, 1, 1))
        conv5_1 = self.conv5_1(conv5_1_pad)

        return [conv1_1, conv2_1, conv3_1, conv4_1, conv5_1]












# class DownSample_1(nn.Module):
#     def __init__(self, in_channel, out_channel, kernel_size=1, stride=1, padding=1):
#         super(DownSample_1, self).__init__()
#
#         self.conv = nn.Conv2d(in_channels=in_channel,
#                               out_channels=out_channel,
#                               kernel_size=kernel_size,
#                               stride=stride,
#                               padding=padding
#                               )
#
#     def forward(self, input_layer):
#         x = input_layer
#
#         out = self.conv(x)
#
#         return out
#
# class DownSample_3(nn.Module):
#     def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
#         super(DownSample_3, self).__init__()
#
#         self.conv = nn.Conv2d(in_channels=in_channel,
#                               out_channels=out_channel,
#                               kernel_size=kernel_size,
#                               stride=stride,
#                               padding=padding
#                               )
#
#     def forward(self, input_layer):
#         x = input_layer
#
#         out = self.conv(x)
#
#         return out
#
#
#
# class Conv2D3(nn.Module):
#     def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
#         super(Conv2D3, self).__init__()
#
#         self.conv = nn.Conv2d(in_channels=in_channel,
#                               out_channels=out_channel,
#                               kernel_size=kernel_size,
#                               stride=stride,
#                               padding=padding
#                               )
#
#     def forward(self, input_layer):
#         x = input_layer
#
#         out = self.conv(x)
#
#         return out
#
#
# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#
#         self.conv1 = Conv2D3(inplanes, planes, stride=stride)
#         self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         self.relu = nn.ReLU(inplace=False)
#
#         self.conv2 = Conv2D3(planes, planes, stride=stride)
#         self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#
#         x = self.conv2(x)
#         x = self.bn2(x)
#
#         if self.downsample is not None:
#             residual = self.downsample(residual)
#
#         out = x + residual
#         out = self.relu(out)
#
#         return out
#
#
# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None):# inplanes=expansion*planes
#         super(Bottleneck, self).__init__()
#
#         self.conv1 = nn.Conv2d(in_channels=inplanes,
#                                out_channels=planes,
#                                kernel_size=1,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#         self.relu = nn.ReLU(inplace=False)
#
#         self.conv2 = nn.Conv2D3(planes, planes, stride=stride)
#         self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
#
#         self.conv3 = nn.Conv2d(in_channels=planes,
#                                out_channels=planes*self.expansion,
#                                kernel_size=1,
#                                bias=False)
#         self.bn3 = nn.BatchNorm2d(planes*self.expansion, momentum=BN_MOMENTUM)
#
#         self.downsample =downsample
#
#     def forward(self, x):
#         residual = x
#
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#
#         x = self.conv3(x)
#         x = self.bn3(x)
#
#         if self.downsample is not None:
#             residual = self.downsample(residual)
#
#         out = x + residual
#         out = self.relu(out)
#
#         return out
#
#
# class HighResolutionBranchStage(nn.Module):
#     def __init__(self, blocks, num_branches, num_blocks, num_inchannels, num_channels,
#                  multi_scale_output=True):
#         super(HighResolutionBranchStage, self).__init__()
#
#         self._check_branches(num_branches, num_blocks, num_inchannels, num_channels)
#         self.blocks = blocks
#         self.num_branches = num_branches
#         self.num_blocks = num_blocks
#         self.num_inchannels = num_inchannels
#         self.num_channels = num_channels
#         self.multi_scale_output = multi_scale_output
#         self.branches = self._make_branches(stride=1)
#
#
#     def _check_branches(self,num_branches, num_blocks, num_inchannels, num_channels):
#         if num_branches != len(num_blocks):
#             print("NEED TO MATCH")
#         if num_branches != len(num_inchannels):
#             print("NEED TO MATCH")
#         if num_branches != len(num_channels):
#             print("NEED TO MATCH")
#
#     def _make_one_branch(self, branch_idx, stride=1):
#         layers = []
#         downsample = False
#         if stride != 1 or \
#         self.num_inchannels[branch_idx] > self.num_channels[branch_idx]*self.blocks.expansion:
#             downsample = True
#         if downsample:
#             downsample = DownSample_1(self.num_inchannel, self.num_channels, stride=stride)
#         for block_n in range(self.num_blocks[branch_idx]):
#             layer = self.blocks(
#                 self.num_inchannels[branch_idx],
#                 self.num_channels[branch_idx],
#                 stride=1,
#                 downsample=downsample
#             )
#             layers.append(layer)
#         out = nn.Sequential(*layers)
#         return out
#
#     def _make_branches(self, stride=1):
#         branch_list = []
#         for branch in range(self.num_branches):
#             branch_list.append(
#                 self._make_one_branch(branch, stride=stride)
#             )
#         out = nn.ModuleList(branch_list)
#         return out
#
#     def forward(self, x):
#         '''
#         :param x: len(x) = self.num_branches
#         :return:
#         '''
#         if self.num_branches == 1:
#             return [self.num_branches[0](x[0])]
#         else:
#             for branch in range(self.num_branches):
#                 x[branch] = self.branches[branch](x[branch])
#         return x
#
#
# class HighResolutionStageFusion(nn.Module):
#     def __init__(self, blocks, branch, num_branches, num_blocks, num_inchannels, num_channels,
#                  multi_scale_output=True):
#         super(HighResolutionStageFusion, self).__init__()
#
#         self._check_branches(num_branches, num_blocks, num_inchannels, num_channels)
#         self.blocks = blocks
#         self.num_branches = num_branches
#         self.num_blocks = num_blocks
#         self.num_inchannels = num_inchannels
#         self.num_channels = num_channels
#         self.multi_scale_output = multi_scale_output
#         self.branches = branch
#         self.fusion_layer_list = self._make_fusion_layer(self.branches)
#
#     def _make_fusion_layer(self, branches):
#         fusion_layers = []
#
#         for branches_out in range(self.num_branches if self.multi_scale_output else 1):
#             tmp_fusion_layers = []
#             for branches_in in range(self.num_branches):
#                 if branches_out == branches_in:
#                     tmp_fusion_layers.append(
#                         branches[branches_in]
#                     )
#                 elif branches_out < branches_in:
#                     tmp_fusion_layers.append(
#                         nn.Sequential(
#                             nn.Conv2d(
#                                 self.num_channels[branches_in],
#                                 self.num_channels[branches_out],
#                                 kernel_size=1,
#                                 stride=1,
#                                 padding=1
#                             ),
#                             nn.BatchNorm2d(num_features=self.num_channels[branches_out],eps=BN_MOMENTUM),
#                             nn.Upsample(scale_factor=2**(branches_in-branches_out), mode='nearest')
#                         )
#                     )
#                 else:
#                     layer_down =[]
#                     for i in range(branches_out-branches_in):
#                         if i == branches_out-branches_in-1:
#                             layer_down.append(
#                                 nn.Sequential(
#                                     nn.Conv2d(
#                                         self.num_channels[branches_in],
#                                         self.num_channels[branches_out],
#                                         kernel_size=3,
#                                         stride=2,
#                                         padding=1
#                                     ),
#                                     nn.BatchNorm2d(num_features=self.num_channels[branches_out],eps=BN_MOMENTUM)
#                                 )
#                             ),
#                         else:
#                             layer_down.append(
#                                 nn.Sequential(
#                                     nn.Conv2d(
#                                         self.num_channels[branches_in],
#                                         self.num_channels[branches_out],
#                                         kernel_size=3,
#                                         stride=2,
#                                         padding=1
#                                     ),
#                                     nn.BatchNorm2d(num_features=self.num_channels[branches_out],eps=BN_MOMENTUM),
#                                     nn.ReLU()
#                                 )
#                             )
#                     tmp_fusion_layers.append(nn.Sequential(*layer_down))
#             fusion_layers.append(nn.ModuleList(tmp_fusion_layers))
#         return nn.ModuleList(fusion_layers)
#
#     def forward(self, x):
#         if self.num_blocks == 1:
#             return [self.branches[0](x[0])]
#         else:
#             for branch in range(self.num_branches):
#                 for sub_branch in range(self.num_branches):
#                     x[branch] += self.fusion_layer_list[branch][sub_branch](x[sub_branch])
#                 x[branch] = nn.ReLU(x[branch])
#
#
#     class HighResolutionTransferStage(nn.Module):
#         pass
#

