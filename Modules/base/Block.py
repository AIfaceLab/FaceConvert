import torch
import torch.nn as nn
import numpy as np


BN_MOMENTUM = 0.01


class DownSample_1(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1, padding=1):
        super(DownSample_1, self).__init__()

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

class DownSample_3(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super(DownSample_3, self).__init__()

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


class Conv2D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super(Conv2D, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channel,
                              out_channels=out_channel,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding
                              )
        # self.avg_pool2d = nn.AvgPool2d(2)

    def forward(self, input_layer):
        x = input_layer

        out = self.conv(x)
        # out = self.avg_pool2d(out)

        return out





class Conv2D_2(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super(Conv2D_2, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channel,
                              out_channels=out_channel,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding
                              )
        self.ReLU = nn.ReLU()

    def forward(self, input_layer):
        x = input_layer

        out = self.conv(x)
        out = self.ReLU(out)

        return out


class Dconv2D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super(Dconv2D, self).__init__()

        self.dconv = nn.ConvTranspose2d(in_channels=in_channel,
                                        out_channels=out_channel,
                                        kernel_size=kernel_size,
                                        stride=stride)

    def forward(self,x):
        out = self.dconv(x)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = Conv2D3(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=False)

        self.conv2 = Conv2D3(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = x + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):# inplanes=expansion*planes
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=inplanes,
                               out_channels=planes,
                               kernel_size=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=False)

        self.conv2 = nn.Conv2D3(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)

        self.conv3 = nn.Conv2d(in_channels=planes,
                               out_channels=planes*self.expansion,
                               kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion, momentum=BN_MOMENTUM)

        self.downsample =downsample

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = x + residual
        out = self.relu(out)

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


class ResBlockDown(nn.Module):
    def __init__(self, in_channel, out_channel, conv_size=3, padding_size=1):
        super(ResBlockDown, self).__init__()

        self.relu = nn.ReLU(inplace=False)
        self.avg_pool2d = nn.AvgPool2d(2)

        # left
        self.conv_l1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 1, ))

        # right
        self.conv_r1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, conv_size, padding=padding_size))
        self.conv_r2 = nn.utils.spectral_norm(nn.Conv2d(out_channel, out_channel, conv_size, padding=padding_size))

    def forward(self, x):
        res = x

        # left
        out_res = self.conv_l1(res)
        out_res = self.avg_pool2d(out_res)

        # right
        out = self.relu(x)
        out = self.conv_r1(out)
        out = self.relu(out)
        out = self.conv_r2(out)
        out = self.avg_pool2d(out)

        # merge
        out = out_res + out

        return out


def adaIN(feature, mean_style, std_style, eps=1e-5):
    B, C, H, W = feature.shape

    feature = feature.view(B, C, -1)

    std_feat = (torch.std(feature, dim=2) + eps).view(B, C, 1)
    mean_feat = torch.mean(feature, dim=2).view(B, C, 1)

    adain = std_style * (feature - mean_feat) / std_feat + mean_style

    adain = adain.view(B, C, H, W)
    return adain


class ResBlockUp(nn.Module):
    def __init__(self, in_channel, out_channel, out_size=None, scale=2, conv_size=3, padding_size=1):
        super(ResBlockUp, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.upsample = nn.Upsample(size=out_size, scale_factor=scale)
        self.relu = nn.ReLU(inplace=False)

        # left
        self.conv_l1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 1))

        # right
        self.conv_r1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, conv_size, padding=padding_size))
        self.conv_r2 = nn.utils.spectral_norm(nn.Conv2d(out_channel, out_channel, conv_size, padding=padding_size))

    def forward(self, x, psi_slice):
        mean1 = psi_slice[:, 0:self.in_channel, :]
        std1 = psi_slice[:, self.in_channel:2 * self.in_channel, :]
        mean2 = psi_slice[:, 2 * self.in_channel:2 * self.in_channel + self.out_channel, :]
        std2 = psi_slice[:, 2 * self.in_channel + self.out_channel: 2 * (self.in_channel + self.out_channel), :]

        res = x

        # left
        out_res = self.upsample(res)
        out_res = self.conv_l1(out_res)

        # right
        # out = adaIN(x, mean1, std1)
        out = self.relu(x)
        out = self.upsample(out)
        out = self.conv_r1(out)
        # out = adaIN(out, mean2, std2)
        out = self.relu(out)
        out = self.conv_r2(out)

        out = out + out_res

        return out


# class HighResolutionStage(nn.Module):
#     def __init__(self, blocks, num_branches, num_blocks, num_inchannels, num_channels,
#                  multi_scale_output=True):
#         super(HighResolutionStage, self).__init__()
#
#         self._check_branches(num_branches, num_blocks, num_inchannels, num_channels)
#         self.blocks = blocks
#         self.num_branches = num_branches
#         self.num_blocks = num_blocks
#         self.num_inchannels = num_inchannels
#         self.num_channels = num_channels
#         self.multi_scale_output = multi_scale_output
#         self.branches = self._make_branches(stride=1)
#         self.fusion_layer_list = self._make_fusion_layer(self.branches)
#
#     def _check_branches(self, num_branches, num_blocks, num_inchannels, num_channels):
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
#         '''
#         :param x: len(x) = self.num_branches
#         :return:
#         '''
#         if self.num_branches == 1:
#             return [self.num_branches[0](x[0])]
#         else:
#             for branch in range(self.num_branches):
#                 x[branch] = self.branches[branch](x[branch])
#             for branch in range(self.num_branches):
#                 for sub_branch in range(self.num_branches):
#                     x[branch] += self.fusion_layer_list[branch][sub_branch](x[sub_branch])
#                 x[branch] = nn.ReLU(x[branch])
#         return x
#
#
# if __name__ == '__main__':
#     blocks = BasicBlock
#     num_branches = 4
#     num_blocks = [4,4,4,4]
#     num_inchannels = [32,64,128,256]
#     num_channels = [32,64,128,256]
#
#     a = HighResolutionStage(blocks=blocks,
#                             num_branches=num_branches,
#                             num_blocks=num_blocks,
#                             num_inchannels=num_inchannels,
#                             num_channels=num_channels,
#                             multi_scale_output=True)
#     for i,m in enumerate(a.modules()):
#         print(i,'->',m)





