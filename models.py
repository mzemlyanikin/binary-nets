from functools import partial

from basic_blocks import BinaryBasicBlock, ParallelBinaryBasicBlock, ParallelBinaryBasicBlockWithFusionGate
from resnet import ResNet

def binary_resnet18(**kwargs):
    model = ResNet(BinaryBasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def parallel_resnet18(**kwargs):
    model = ResNet(partial(ParallelBinaryBasicBlock, squeeze=False), [2, 2, 2, 2], **kwargs)
    return model


def yet_another_parallel_resnet18(**kwargs):
    model = ResNet(partial(BinaryBasicBlock, squeeze=True), [2, 2, 2, 2], **kwargs)
    return model


def parallel_resnet18_with_fusion_gate(**kwargs):
    model = ResNet(BinaryBasicBlockWithFusionGate, [2, 2, 2, 2], **kwargs)
    return model

