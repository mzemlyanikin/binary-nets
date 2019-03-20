import torch
import torch.nn as nn

import numpy as np

from modules import binary_conv3x3


class ParallelBinaryBasicBlockWithFusionGate(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, parallel=1, multiplication=False, downsample=None, **kwargs):
        super(ParallelBinaryBasicBlockWithFusionGate, self).__init__()
        inplanes = inplanes * parallel
        planes = planes * parallel
        self.planes = planes
        self.conv1 = binary_conv3x3(inplanes, planes, stride, groups=parallel, **kwargs)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)#nn.Hardtanh(inplace=True)#
        self.conv2 = binary_conv3x3(planes, planes, groups=parallel, **kwargs)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.parallel = parallel
        if multiplication:
            self.lambdas = nn.ParameterList([nn.Parameter(torch.ones([1], dtype=torch.float32)) for _ in range(parallel)]) # fix is needed
        self.multiplication = multiplication
        self.C_s = nn.ParameterList([nn.Parameter(torch.tensor(np.array([0]), dtype=torch.float32)) for _ in range(parallel)]) # what are the good init parameters?

    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        num_channels = self.planes

        if self.multiplication:
            multiplied = []
            for idx, lambda_i in enumerate(self.lambdas):
                start_ch = num_channels * idx // self.parallel
                end_ch = num_channels * (idx + 1) // self.parallel
                multiplied.append(out[:, start_ch:end_ch, :, :] * lambda_i)
            out = torch.cat(multiplied, 1)

        if self.parallel > 1:
            squeezed = out.view(out.shape[0], self.parallel, out.shape[1] // self.parallel, out.shape[2], out.shape[3]).sum(dim=1).squeeze(1)
            c_multiplied = []
            for C in self.C_s:
                C = torch.clamp(C, 0, 1)
                start_ch = num_channels * idx // self.parallel
                end_ch = num_channels * (idx + 1) // self.parallel
                c_multiplied.append(out[:, start_ch:end_ch, :, :] * C + squeezed * (1 - C))
            out = torch.cat(c_multiplied, 1)

        out += identity
        out = self.relu(out)
        return out
