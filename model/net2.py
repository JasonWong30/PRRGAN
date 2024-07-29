import torch.nn as nn
from model.ResBlock import ResBlock
import torch
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
import numpy as np
from kornia.losses import ssim_loss as ssim

class MODEL(nn.Module):
    def __init__(self):
        super(MODEL, self).__init__()
        self.layer = Block()

    def forward(self, other, color):
        loss_ssim, output = self.layer(other, color)
        # y = torch.unsqueeze(y,dim=1)
        return  loss_ssim, output


class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        img_channel = 1
        out_channels = 1
        num_featuers = 8
        self.layer1 = ResBlock(img_channel, out_channels, num_featuers, stride=1, groups=1, norm_layer=None,
                               doublea=True)
        self.sigmoid = nn.Sigmoid()

    def forward_once(self, x):
        output = self.layer1(x)
        # output = self.layer2(output)
        return output

    def _softmax(self, tensor):
        """
        A private method which compute softmax ouput of a given tensor
        """
        tensor = torch.exp(tensor)
        tensor = tensor / tensor.sum(dim=1, keepdim=True)
        return tensor

    def forward_loss(self, fused_img, color, other):
        loss1 =  ssim(fused_img, color, 11)
        loss2 =  ssim(fused_img, other, 11)
        ssim_loss = loss1 + loss2
        return ssim_loss

    # self, input1, input2, input3
    def forward(self, input1, input2):
        tensor1 = self.forward_once(input1)
        tensor2 = self.forward_once(input2)
        k1 = tensor1 ** 2 / (tensor1 ** 2 + tensor2 ** 2)
        k2 = tensor2 ** 2 / (tensor1 ** 2 + tensor2 ** 2)
        output = k1 * input1 + k2 * input2

        loss_ssim = self.forward_loss(output, input1, input2)

        return loss_ssim, output
