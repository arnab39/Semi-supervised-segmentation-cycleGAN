from torch import nn
from .ops import conv_bn_lrelu
import torch

class Discriminator(nn.Module):
    def __init__(self, in_dim=3, inter_dim=64):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(nn.Conv2d(in_dim, inter_dim, 4, 2, 1), nn.LeakyReLU(0.2),
                                 conv_bn_lrelu(inter_dim * 1, inter_dim * 2, 4, 2, 1),
                                 conv_bn_lrelu(inter_dim * 2, inter_dim * 4, 4, 2, 1),
                                 conv_bn_lrelu(inter_dim * 4, inter_dim * 8, 4, 1, (1, 2)),
                                 nn.Conv2d(inter_dim * 8, 1, 4, 1, (2, 1)))

    def forward(self, x):
        return self.dis(x)

## define your discriminators here



