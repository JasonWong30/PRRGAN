import os
import torch
import torch.nn as nn
from model.SENet import DoubleAtten
import torch.nn.functional as F
import math
from model.double_attention_layer import DoubleAttentionLayer

def conv3x3(in_channels, out_channels, stride=1, groups=1, bias=False):
    """3x3 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=bias)


def conv1x1(in_channels, out_channels, stride=1, groups=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, groups=groups, bias=bias)

class MSB(nn.Module):
    def __init__(self, bottleneck_planes, channels_scale,  stride=1, groups=1, norm_layer=None, scales = 2):
        super(MSB, self).__init__()
        self.scales = scales
        self.modlist0 = nn.ModuleList([conv1x1(bottleneck_planes, channels_scale, stride=stride),
                                  norm_layer(channels_scale),
                                  nn.ReLU(inplace=True),
                                  conv3x3(channels_scale, channels_scale, stride=stride, groups=groups),
                                  norm_layer(channels_scale)])
        self.modlist1 = nn.ModuleList([conv1x1(bottleneck_planes, channels_scale, stride=stride),
                                  norm_layer(channels_scale),
                                  nn.ReLU(inplace=True),
                                  conv3x3(channels_scale, channels_scale, stride=stride, groups=groups),
                                  norm_layer(channels_scale),
                                  nn.ReLU(inplace=True),
                                  conv3x3(channels_scale, channels_scale, stride=stride, groups=groups),
                                  norm_layer(channels_scale)])

    def forward(self, out):
        in_branch0 = out
        in_branch1 = out
        ys = []
        for s in range(self.scales):
            if s == 0:
                for m in self.modlist0:
                    in_branch0 = m(in_branch0)
                ys.append(in_branch0)
            elif s == 1:
                for m in self.modlist1:
                    in_branch1 = m(in_branch1)
                ys.append(in_branch1)
        out = torch.cat(ys, 1)
        return out

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_featuers, stride=1, scales=2, groups=1, norm_layer=None, doublea=True,flag=True):
        super(ResBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d                        # BN layer
        bottleneck_planes = num_featuers
        channels_scale = bottleneck_planes // scales
        self.flag = flag
        self.stride = stride
        self.scales = scales
        self.doublea = doublea
        self.conv1 = conv3x3(in_channels, bottleneck_planes, stride)
        self.bn1 = norm_layer(bottleneck_planes)
        self.relu = nn.ReLU(inplace=True)    # inplace: Overwrite original value
        self.MSB = MSB(bottleneck_planes, channels_scale,  stride=1, groups=1, norm_layer=norm_layer, scales = self.scales)
        self.doubleatt = DoubleAttentionLayer(bottleneck_planes, bottleneck_planes, bottleneck_planes, bottleneck_planes, reconstruct = True) if doublea else None

        self.conv2 = conv1x1(bottleneck_planes, out_channels, stride=stride)
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        identy = out
        out = self.MSB(out)
        if self.doubleatt is not None:
            out = self.doubleatt(out)
        if self.flag:
            out = self.conv2(out)
        else:
            out= out + identy
        return out


