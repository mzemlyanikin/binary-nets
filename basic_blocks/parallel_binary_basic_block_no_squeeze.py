import torch.nn as nn


class ParallelBinaryBasicBlock(nn.Module): # paper, page 5, fig. 2 (c)
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, parallel=1, multiplication=False, downsample=None):
        super(ParallelBinaryBasicBlock, self).__init__()
        inplanes = inplanes * parallel
        planes = planes * parallel
        self.conv1 = binary_conv3x3(inplanes, planes, stride, groups=parallel)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)#nn.Hardtanh(inplace=True)#
        self.conv2 = binary_conv3x3(planes, planes, groups=parallel)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.parallel = parallel
        if multiplication:
            self.lambdas = nn.Parameter([nn.Parameter(torch.ones([1], dtype=torch.float32)) for _ in range(parallel)]) # fix is needed
        self.multiplication = multiplication

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

        if self.multiplication:
            tmp_shape = list(out.size())
            tmp_shape[1] = out.size()[1] // self.parallel
            tensors = []
            for lambda_i in self.lambdas:
                tensors.append(lambda_i.expand(*tmp_shape))
            lambdas = torch.cat(tensors, 1)
            tmp = out * lambdas
            return tmp
        else:
            return out

