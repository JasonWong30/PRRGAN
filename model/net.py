import torch.nn as nn
from model.ResBlock import ResBlock
import torch
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
import numpy as np
EPSILON = 1e-3
# from losses import ssim
from kornia.losses import ssim_loss as ssim
# from skimage.metrics import structural_similarity as ssim

class MODEL(nn.Module):
    def __init__(self):
        super(MODEL, self).__init__()
        self.layer = Block()
    # other, color / color, T2, other
    def forward(self, color, other, T2):
        y = self.layer(color, other, T2)   #11111
        # y = torch.unsqueeze(y,dim=1)
        return y


class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()

        img_channel = 1
        out_channel = 1
        num_featuers = 32
        self.layer1 = ResBlock(img_channel, num_featuers, num_featuers, stride=1, groups=1, norm_layer=None,
                               doublea=True,flag=False)
        self.layer2 = ResBlock(num_featuers, out_channel, num_featuers, stride=1, groups=1, norm_layer=None,
                               doublea=True,flag=True)

    def attention_fusion_weight(self,tensor1, tensor2, tensor3):
        # avg, max, nuclear
        tmp = self.spatial_fusion(tensor1, tensor2, tensor3)

        return tmp

    def forward_once(self, x):
        output = self.layer1(x)
        output = self.layer2(output)

        return output

    def make_mask(self,input):
        mask = (input == input.max(dim=1, keepdim=True)[0]).to(dtype=torch.int32)
        return mask

    # self, tensor1, tensor2, tensor3
    def tensor_max(self, tensor1, tensor2):
        # max_tensor = torch.max(tensor1, tensor2,tensor3)
        max_tensor = torch.max(tensor1, tensor2)
        return max_tensor

    def tensor_cat(self, tensor1, tensor2):
        cat_tensor = torch.cat((tensor1, tensor2), 1)
        return cat_tensor

    def _softmax(self, tensor):
        """
        A private method which compute softmax ouput of a given tensor
        """
        tensor = torch.exp(tensor)
        tensor = tensor / tensor.sum(dim=1, keepdim=True)
        return tensor

    def forward_loss(self, fused_img, color, other, T2):
        loss1 =  ssim(fused_img, color, 11)
        loss2 =  ssim(fused_img, other, 11)
        loss3 =  ssim(fused_img, T2, 11)
        ssim_loss = loss1 + loss2 + loss3
        return ssim_loss

    # self, input1, input2, input3 ; color, T2, other
    def forward(self, input1, input2, input3):
        tensor1= self.forward_once(input1)
        tensor2 = self.forward_once(input2)
        tensor3 = self.forward_once(input3)
        #权重
        k1 = tensor1 ** 2 / (tensor1 ** 2 + tensor2 ** 2 + tensor3 ** 2)
        k2 = tensor2 ** 2 / (tensor1 ** 2 + tensor2 ** 2 + tensor3 ** 2)
        k3 = tensor3 ** 2 / (tensor1 ** 2 + tensor2 ** 2 + tensor3 ** 2)

        output = k1 * input1 + k2 * input2 +  + k3 * input3
        loss_ssim = self.forward_loss(output, input1, input2, input3)

        return loss_ssim, output