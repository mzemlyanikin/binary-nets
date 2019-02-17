import torch.nn as nn

from modules import binary_conv3x3


class ParallelBinaryBasicBlock(nn.Module):
    expansion = 1
    # paper, page 5, fig. 2 (b) if squeeze = True
    # paper, page 5, fig. 2 (c) if squeeze = False

    def __init__(self, inplanes, planes, stride=1, parallel=1, multiplication=False, downsample=None, squeeze=False):
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
            self.lambdas = nn.ParameterList([nn.Parameter(torch.ones([1], dtype=torch.float32)) for _ in range(parallel)]) # fix is needed
        self.multiplication = multiplication
        self.squeeze = squeeze

    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.squeeze and self.parallel != 1:
            x = x.repeat(1, self.parallel, 1, 1)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.multiplication:
            tmp_shape = list(out.size())
            tmp_shape[1] = out.size()[1] // self.parallel
            tensors = []
            for lambda_i in self.lambdas:
                tensors.append(lambda_i.expand(*tmp_shape))
            lambdas = torch.cat(tensors, 1)
            out = out * lambdas

        if self.squeeze and self.parallel > 1:
            x_shape = out.size()
            x_unsq = out.unsqueeze(1)
            x_resh = x_unsq.reshape(x_shape[0], self.parallel, x_shape[1] // self.parallel, x_shape[2],
                                    x_shape[3])  # is it OK?
            x_sum = x_resh.sum(dim=1)
            out = x_sum.squeeze(1)

        out += identity # Here all tricks were made before residual connection and last relu
        out = self.relu(out)
        return out

