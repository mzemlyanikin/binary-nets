import torch.nn as nn
from basic_blocks import ParallelBinaryBasicBlockNoSqueeze, ParallelBinaryBasicBlockWithSqueeze, ParallelBinaryBasicBlockWithFusionGate

from modules import conv1x1


class ResNet(nn.Module):

    def __init__(self, block, layers, parallel=1, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.block = block
        self.parallel = parallel
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True) #nn.Hardtanh(inplace=True)#
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], parallel=parallel)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, parallel=parallel)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, parallel=parallel)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, parallel=parallel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

       #  for m in self.modules():
       #      if isinstance(m, nn.Conv2d):
       #          nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
       #      elif isinstance(m, nn.BatchNorm2d):
       #          nn.init.constant_(m.weight, 1)
       #          nn.init.constant_(m.bias, 0)

       #  # Zero-initialize the last BN in each residual branch,
       #  # so that the residual branch starts with zeros, and each residual block behaves like an identity.
       #  # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
       #  if zero_init_residual:
       #      for m in self.modules():
       #          if isinstance(m, BinaryBottleneck):
       #              nn.init.constant_(m.bn3.weight, 0)
       #          elif isinstance(m, BinaryBasicBlock):
       #              nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, parallel=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if block in [ParallelBinaryBasicBlockNoSqueeze, ParallelBinaryBasicBlockWithFusionGate]:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes * parallel, planes * block.expansion * parallel, stride, parallel), # is it OK?
                    nn.BatchNorm2d(planes * block.expansion * parallel),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    # is it OK?
                    nn.BatchNorm2d(planes * block.expansion),
                )

        layers = []
        if block in [ParallelBinaryBasicBlockNoSqueeze, ParallelBinaryBasicBlockWithSqueeze, ParallelBinaryBasicBlockWithFusionGate]:
            appended_layer = block(self.inplanes, planes, stride=stride, downsample=downsample, parallel=parallel, multiplication=True)
        else:
            appended_layer = block(self.inplanes, planes, stride=stride, downsample=downsample)
        layers.append(appended_layer)
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            if block in [ParallelBinaryBasicBlockNoSqueeze, ParallelBinaryBasicBlockWithSqueeze, ParallelBinaryBasicBlockWithFusionGate]:
                layers.append(block(self.inplanes, planes, parallel=parallel, multiplication=True))
            else:
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x) # Pooling is skipped for CIFAR

        if self.parallel != 1 and self.block != ParallelBinaryBasicBlockWithSqueeze:
            x = x.repeat(1, self.parallel, 1, 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.block in [ParallelBinaryBasicBlockNoSqueeze, ParallelBinaryBasicBlockWithFusionGate]:
            x_shape = x.size()
            x_unsq = x.unsqueeze(1)
            x_resh = x_unsq.reshape(x_shape[0], self.parallel, x_shape[1] // self.parallel, x_shape[2], x_shape[3]) # is it OK?
            x_sum = x_resh.sum(dim=1)
            x = x_sum.squeeze(1)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

