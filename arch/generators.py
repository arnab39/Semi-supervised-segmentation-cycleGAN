from torch import nn
from .ops import conv_bn_lrelu, conv_bn_relu, dconv_bn_relu, ResidualBlock


class Generator(nn.Module):
    def __init__(self, in_dim=3, inter_dim=64, out_dim=3):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(nn.ReflectionPad2d(3),
                                 conv_bn_relu(in_dim, inter_dim * 1, 7, 1),
                                 conv_bn_relu(inter_dim * 1, inter_dim * 2, 3, 2, 1),
                                 conv_bn_relu(inter_dim * 2, inter_dim * 4, 3, 2, 1),
                                 ResidualBlock(inter_dim * 4, inter_dim * 4),
                                 ResidualBlock(inter_dim * 4, inter_dim * 4),
                                 ResidualBlock(inter_dim * 4, inter_dim * 4),
                                 ResidualBlock(inter_dim * 4, inter_dim * 4),
                                 ResidualBlock(inter_dim * 4, inter_dim * 4),
                                 ResidualBlock(inter_dim * 4, inter_dim * 4),
                                 ResidualBlock(inter_dim * 4, inter_dim * 4),
                                 ResidualBlock(inter_dim * 4, inter_dim * 4),
                                 ResidualBlock(inter_dim * 4, inter_dim * 4),
                                 dconv_bn_relu(inter_dim * 4, inter_dim * 2, 3, 2, 1, 1),
                                 dconv_bn_relu(inter_dim * 2, inter_dim * 1, 3, 2, 1, 1),
                                 nn.ReflectionPad2d(3),
                                 nn.Conv2d(inter_dim, out_dim, 7, 1),
                                 nn.Tanh())

    def forward(self, x):
        return self.gen(x)
# define your self-define generators here
