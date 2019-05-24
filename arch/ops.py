import functools
from torch.nn import init
import torch.nn as nn
import torch

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.normal(m.weight.data, 0.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal(m.weight.data, 1.0, gain)
            init.constant(m.bias.data, 0.0)

    print('Network initialized with weights sampled from N(0,0.02).')
    net.apply(init_func)


def init_network(net, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net)
    return net


def conv_norm_lrelu(in_dim, out_dim, kernel_size, stride = 1, padding=0,
                                 norm_layer = nn.BatchNorm2d, bias = False):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias = bias),
        norm_layer(out_dim), nn.LeakyReLU(0.2,True))

def conv_norm_relu(in_dim, out_dim, kernel_size, stride = 1, padding=0,
                                 norm_layer = nn.BatchNorm2d, bias = False):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias = bias),
        norm_layer(out_dim), nn.ReLU(True))

def dconv_norm_relu(in_dim, out_dim, kernel_size, stride = 1, padding=0, output_padding=0,
                                 norm_layer = nn.BatchNorm2d, bias = False):
    return nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride,
                           padding, output_padding, bias = bias),
        norm_layer(out_dim), nn.ReLU(True))


class ResidualBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_dropout, use_bias):
        super(ResidualBlock, self).__init__()
        res_block = [nn.ReflectionPad2d(1),
                     conv_norm_relu(dim, dim, kernel_size=3, 
                     norm_layer= norm_layer, bias=use_bias)]
        if use_dropout:
            res_block += [nn.Dropout(0.5)]
        res_block += [nn.ReflectionPad2d(1),
                      nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
                      norm_layer(dim)]

        self.res_block = nn.Sequential(*res_block)

    def forward(self, x):
        return x + self.res_block(x)


def set_grad(nets, requires_grad=False):
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad


 #######
 
### A huge thanks to the repo: https://github.com/davidtvs/PyTorch-ENet
### A lot of ideas are being taken up from this repo

'''
Here we start off for writing the required blocks of code for the enet model
'''

class InitialBlock(nn.Module):
    '''
    This will be initial block of the network
    '''

    def __init__(self, input_dim = 3, output_dim = 16, kernel_size = 3, relu = True, padding=0, bias=False):
        super().__init__()

        if relu:
            activation = nn.ReLU()    
        else:
            activation = nn.PReLU()
        
        ### So here we will define the main branch which will give the output by passing through the conv block
        self.main_branch = nn.Conv2d(in_channels = input_dim,
                                     out_channels = output_dim - 3,
                                     kernel_size = kernel_size,
                                     stride= 2,
                                     padding= padding,
                                     bias = bias)
        
        ### Extension branch, which is basically MaxPool2d in this case
        self.ext_branch = nn.MaxPool2d(kernel_size = kernel_size,
                                       stride= 2,
                                       padding= padding)

        ### Some of the extra basic blocks of activations and other things
        self.batch_norm = nn.BatchNorm2d(output_dim)
        self.activation = activation

    def forward(self, x):
        main = self.main_branch(x)
        ext = self.ext_branch(x)

        out = torch.cat((main, ext), 1)   ### For concatenating their outputs

        out = self.batch_norm(out)
        out = self.activation(out)

        return out

class RegularBottleneck(nn.Module):
    '''
    So in this also there are two branches:
    Main branch -->
        This is basically simply the input
    Extension Branch -->
        1. 1*1 conv to decrease the spatial dimensions to a internal_ratio specified by the user
        2. conv layer, which can be regular, atrous or asymmetric convolution
        3. 1*1 to increase the spatial dimension back from internal_ratio to the original size
        4. Regularizer which in this case is dropout
    '''

    def __init__(self, channels, internal_ratio = 4, kernel_size = 3, dilation = 1, padding = 1, asymmetric = False, dropout_prob=0, bias=False, relu=True):
        super().__init__()

        ### Now we need to ensure if the internal_ratio has a reasonable value
        assert internal_ratio > 1 and internal_ratio < channels , 'The value of the internal_ratio is not correct'

        internal_channels = channels // internal_ratio

        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        self.ext_conv1 = nn.Sequential(
                         nn.Conv2d(in_channels = channels, out_channels = internal_channels, kernel_size = 1, stride = 1, padding = 0, bias=bias),
                         nn.BatchNorm2d(internal_channels),
                         activation)

        ### So here we are going to see for the case if we are given asymmetric convolutions
        ### So basically it is gonna be like, example for 3*3 we will break it into 3*1 and 1*3
        ### and then perform normal convolutions, first 3*1 and then 1*3
        if asymmetric:
            self.ext_conv2 = nn.Sequential(
                                nn.Conv2d(internal_channels, internal_channels, (kernel_size, 1), stride = (1, 1), padding = (padding, 0), dilation=dilation, bias=bias),  
                                nn.BatchNorm2d(internal_channels),
                                activation,
                                nn.Conv2d(internal_channels, internal_channels, kernel_size = (1, kernel_size), stride=(1,1), padding=(0, padding), dilation=dilation, bias=bias),
                                nn.BatchNorm2d(internal_channels),
                                activation
            )
        else:
            self.ext_conv2 = nn.Sequential(
                                nn.Conv2d(internal_channels, internal_channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=bias),
                                nn.BatchNorm2d(internal_channels),
                                activation
            )
        
        ### Now we are gonna implement the 1*1 expansion code
        self.ext_conv3 = nn.Sequential(nn.Conv2d(internal_channels, channels, kernel_size=1, stride=1, padding=0, bias=bias),
                                       nn.BatchNorm2d(channels),
                                       activation)
        
        self.ext_regularizer = nn.Dropout2d(p = dropout_prob)

        self.out_relu = activation

    
    def forward(self, x):

        main = x

        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regularizer(ext)

        out = ext + main

        ### Activation applying after the addition
        return self.out_relu(out)


class DownsamplingBottleneck(nn.Module):
    '''
    This is the Downsampling bottleneck layer to downsample the input
    '''

    def __init__(self, in_channels, out_channels, internal_ratio=4, kernel_size = 3, padding=0, return_indices = False, dropout_prob = 0, bias=False, relu = True):
        super().__init__()

        self.return_indices = return_indices

        ### Now we need to ensure if the internal_ratio has a reasonable value
        assert internal_ratio > 1 and internal_ratio < in_channels , 'The value of the internal_ratio is not correct'

        internal_channels = in_channels // internal_ratio

        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        self.main_max1 = nn.MaxPool2d(kernel_size = kernel_size, stride=2, padding=padding, return_indices=return_indices)

        ### The extension branch
        self.ext_conv1 = nn.Sequential(
                            nn.Conv2d(in_channels, internal_channels, kernel_size=2, stride=2, bias=bias),
                            nn.BatchNorm2d(internal_channels),
                            activation)
            
        ### Now we are gonna do the conv operation
        self.ext_conv2 = nn.Sequential(
                            nn.Conv2d(internal_channels, internal_channels, kernel_size= kernel_size, stride=1, padding=padding, bias=bias),
                            nn.BatchNorm2d(internal_channels),
                            activation
        )

        self.ext_conv3 = nn.Sequential(
                            nn.Conv2d(internal_channels, out_channels, kernel_size=1, stride=1, bias=bias),
                            nn.BatchNorm2d(out_channels),
                            activation
        )

        self.ext_reg = nn.Dropout2d(p=dropout_prob)

        self.out_activation = activation

    def forward(self, x):
        if self.return_indices:
            main, max_indices = self.main_max1(x)
        else:
            main = self.main_max1(x)

        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_reg(ext)

        ### Main branch channel padding
        n, ch_ext, h, w = ext.size()
        ch_main = main.size()[1]
        padding = torch.zeros(n, ch_ext - ch_main, h, w)

        ### So before concatenating we would first have to see if the main is on gpu cause they both must be
        ### on the same memory
        if main.is_cuda:
            padding = padding.cuda()
        

        ### Concatenating
        main = torch.cat((main, padding), 1)

        out = main + ext

        if self.return_indices:
            return self.out_activation(out), max_indices
        else:
            return self.out_activation(out)


class UpsamplingBottleneck(nn.Module):
    '''
    This is the Upsampling bottleneck as is displayed in the paper
    '''

    def __init__(self, in_channels, out_channels, internal_ratio = 4, kernel_size = 3, padding=0, dropout_prob = 0, bias=False, relu=True):
        super().__init__()

        ### Now we need to ensure if the internal_ratio has a reasonable value
        assert internal_ratio > 1 and internal_ratio < in_channels , 'The value of the internal_ratio is not correct'

        internal_channels = in_channels // internal_ratio

        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        self.main_conv1 = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, kernel_size= kernel_size, padding=padding, bias = bias),
                            nn.BatchNorm2d(out_channels)
        )

        self.main_unpool1 = nn.MaxUnpool2d(kernel_size=2)

        ### Extension branch
        self.ext_conv1 = nn.Sequential(
                            nn.Conv2d(in_channels, internal_channels, kernel_size= 1, bias= bias),
                            nn.BatchNorm2d(internal_channels),
                            activation
        )

        self.ext_conv2 = nn.Sequential(
                            nn.ConvTranspose2d(internal_channels, internal_channels, kernel_size=kernel_size, stride=2, padding=padding, output_padding=1, bias=bias),
                            nn.BatchNorm2d(internal_channels),
                            activation
        )

        self.ext_conv3 = nn.Sequential(
                            nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=bias),
                            nn.BatchNorm2d(out_channels),
                            activation
        )

        self.ext_reg = nn.Dropout2d(p=dropout_prob)

        self.out_activation = activation
    
    def forward(self, x, max_indices):
        main = self.main_conv1(x)
        main = self.main_unpool1(main, max_indices)

        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_reg(ext)

        out = main+ext

        return self.out_activation(out)