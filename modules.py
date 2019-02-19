import torch
import torch.nn as nn
import torch.nn.functional as F

from binarizers import WeightBinarizer, ActivationBinarizer, Ternarizer, Identity


def binarize(input): # Simplest possible binarization
    return input.sign()

# Binary Conv2d is taken from:
# https://github.com/itayhubara/BinaryNet.pytorch/blob/master/models/binarized_modules.py
class BinaryConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, activation_binarizer=ActivationBinarizer(), weight_binarizer=WeightBinarizer()):
        super(BinaryConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.binarize_act = activation_binarizer
        self.binarize_w = weight_binarizer

    def forward(self, input):
        if input.size(1) != 3:
            input = self.binarize_act(input)
        if not hasattr(self.weight, 'original'):
            self.weight.original = self.weight.data.clone()
        self.weight.data = self.binarize_w(self.weight.original)
        #self.weight.data = binarize(self.weight.data)
        out = F.conv2d(input, self.weight, None, self.stride,
                        self.padding, self.dilation, self.groups)

        if self.bias is not None:
            # self.bias.original = self.bias.data.clone() # do we need to save bias copy if it's not quantized?
            out += self.bias.view(1, -1, 1, 1).expand_as(out)
        return out


def binary_conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return BinaryConv2d(in_planes, out_planes, kernel_size=3, stride=stride, groups=groups,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1, groups=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, groups=groups, bias=False)


def binary_conv1x1(in_planes, out_planes, stride=1, groups=1):
    """1x1 convolution"""
    return BinaryConv2d(in_planes, out_planes, kernel_size=1, stride=stride, groups=groups, bias=False)

