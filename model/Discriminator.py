import math

import torch.nn.functional as F
from torch import nn
import torch


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),

            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(16, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x1 = self.net(x).view(batch_size,1)
        # return torch.sigmoid(x1)
        return x1