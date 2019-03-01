from basic_blocks import BinaryBasicBlock, ParallelBinaryBasicBlockWithSqueeze, ParallelBinaryBasicBlockNoSqueeze, ParallelBinaryBasicBlockWithFusionGate
from resnet import ResNet, ReIdResNet
from load_weights import load_weights

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def binary_resnet18(pretrained='imagenet', **kwargs): # paper, page 5, fig. 1(b)
    model = ResNet(BinaryBasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained == 'imagenet':
        load_weights(model, model_urls['resnet18'], partial=True)
    return model


def parallel_resnet18_no_squeeze(pretrained='imagenet', **kwargs): # paper, page 5, fig. 2 (c)
    model = ResNet(ParallelBinaryBasicBlockNoSqueeze, [2, 2, 2, 2], **kwargs)
    if pretrained == 'imagenet':
        load_weights(model, model_urls['resnet18'], partial=True)
    return model


def parallel_resnet18_with_squeeze(pretrained='imagenet', **kwargs): # paper, page 5, fig. 2 (b)
    model = ResNet(ParallelBinaryBasicBlockWithSqueeze, [2, 2, 2, 2], **kwargs)
    if pretrained == 'imagenet':
        load_weights(model, model_urls['resnet18'], partial=True)
    return model


def parallel_resnet18_with_fusion_gate(pretrained='imagenet', **kwargs): # paper, page 6, fig. 3
    model = ResNet(ParallelBinaryBasicBlockWithFusionGate, [2, 2, 2, 2], **kwargs)
    if pretrained == 'imagenet':
        load_weights(model, model_urls['resnet18'], partial=True)
    return model


def reid_binary_resnet18(loss, num_classes, pretrained='imagenet', **kwargs): # paper, page 5, fig. 1(b)
    model = ReIdResNet(loss=loss, num_classes=num_classes, block=BinaryBasicBlock, layers=[2, 2, 2, 2], **kwargs)
    if pretrained == 'imagenet':
        load_weights(model, model_urls['resnet18'], partial=True)
    return model


def reid_parallel_resnet18_no_squeeze(loss, num_classes, pretrained='imagenet', **kwargs): # paper, page 5, fig. 2 (c)
    model = ReIdResNet(loss=loss, num_classes=num_classes, block=ParallelBinaryBasicBlockNoSqueeze, layers=[2, 2, 2, 2], **kwargs)
    if pretrained == 'imagenet':
        load_weights(model, model_urls['resnet18'], partial=True)
    return model


def reid_parallel_resnet18_with_squeeze(loss, num_classes, pretrained='imagenet', **kwargs): # paper, page 5, fig. 2 (b)
    model = ReIdResNet(loss=loss, num_classes=num_classes, block=ParallelBinaryBasicBlockWithSqueeze, layers=[2, 2, 2, 2], **kwargs)
    if pretrained == 'imagenet':
        load_weights(model, model_urls['resnet18'], partial=True)
    return model


def reid_parallel_resnet18_with_fusion_gate(loss, num_classes, pretrained='imagenet', **kwargs): # paper, page 6, fig. 3
    model = ReIdResNet(loss=loss, num_classes=num_classes, block=ParallelBinaryBasicBlockWithFusionGate, layers=[2, 2, 2, 2], **kwargs)
    if pretrained == 'imagenet':
        load_weights(model, model_urls['resnet18'], partial=True)
    return model

