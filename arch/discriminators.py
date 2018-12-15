from torch import nn
from .ops import conv_bn_lrelu
import torch
from torch.nn import functional as F


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


class Discriminator2(nn.Module):
    def __init__(self, in_dim):
        super(Discriminator2, self).__init__()

        # A bunch of convolutions one after another
        model = [nn.Conv2d(in_dim, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(256, 512, 4, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=True)]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
