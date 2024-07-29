import torch.nn as nn
import torch
import torch.nn.functional as F
from  GCN_lib.Rs_GCN import GCN

class DoubleAtten(nn.Module):
    """
    A2-Nets: Double Attention Networks. NIPS 2018
    """
    def __init__(self,in_c,embed_size):
        super(DoubleAtten,self).__init__()
        self.in_c = in_c
        """Convolve the same input feature map to produce three feature maps with the same scale, i.e., A, B, V (as shown in paper).
        """
        self.convA = nn.Conv2d(in_c,in_c,kernel_size=1)
        self.convB = nn.Conv2d(in_c,in_c,kernel_size=1)
        self.convV = nn.Conv2d(in_c,in_c,kernel_size=1)
        self.Rs_GCN = GCN(in_channels=embed_size, inter_channels=embed_size)

    def forward(self,input):

        feature_maps = self.convA(input)
        atten_map = self.convB(input)

        b, _, h, w = feature_maps.shape

        feature_maps = feature_maps.view(b, 1, self.in_c, h*w)
        atten_map = atten_map.view(b, self.in_c, 1, h*w)
        global_descriptors = torch.mean((feature_maps * F.softmax(atten_map, dim=-1)),dim=-1) # Multiply the feature map and the attention weight map to generate a global feature descriptor
        global_descriptors_star = self.Rs_GCN(global_descriptors)

        v = self.convV(input)
        atten_vectors = F.softmax(v.view(b, self.in_c, h*w), dim=-1)
        out = torch.bmm(atten_vectors.permute(0,2,1), global_descriptors_star).permute(0,2,1)

        return out.view(b, _, h, w)

