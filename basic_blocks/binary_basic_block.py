import torch.nn as nn
from modules import binary_conv3x3


class BinaryBasicBlock(nn.Module):  # paper, page 5, fig. 1 (b)
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BinaryBasicBlock, self).__init__()
        self.conv1 = binary_conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True) #nn.Hardtanh(inplace=True)#
        self.conv2 = binary_conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

