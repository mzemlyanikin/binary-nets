from basic_blocks import BinaryBasicBlock, ParallelBinaryBasicBlockWithSqueeze, ParallelBinaryBasicBlockNoSqueeze, ParallelBinaryBasicBlockWithFusionGate
from resnet import ResNet

def binary_resnet18(**kwargs): # paper, page 5, fig. 1(b)
    model = ResNet(BinaryBasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def parallel_resnet18_no_squeeze(**kwargs): # paper, page 5, fig. 2 (c)
    model = ResNet(ParallelBinaryBasicBlockNoSqueeze, [2, 2, 2, 2], **kwargs)
    return model


def parallel_resnet18_with_squeeze(**kwargs): # paper, page 5, fig. 2 (b)
    model = ResNet(ParallelBinaryBasicBlockWithSqueeze, [2, 2, 2, 2], **kwargs)
    return model


def parallel_resnet18_with_fusion_gate(**kwargs): # paper, page 6, fig. 3
    model = ResNet(ParallelBinaryBasicBlockWithFusionGate, [2, 2, 2, 2], **kwargs)
    return model

