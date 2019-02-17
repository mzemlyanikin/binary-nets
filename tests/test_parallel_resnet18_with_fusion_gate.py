import pytest
import torch

from resnet import ResNet
from models import parallel_resnet18_with_fusion_gate


class TestParallelResnet18WithFusionGate:

    def test_dummy(self):
        assert(isinstance(parallel_resnet18_with_fusion_gate(), ResNet))

    def test_dummy_parallel2(self):
        assert(isinstance(parallel_resnet18_with_fusion_gate(parallel=2), ResNet))
    
    def test_forward(self):
        model = parallel_resnet18_with_fusion_gate()
        assert(isinstance(model(torch.rand(1, 3, 224, 224)), torch.Tensor))

    def test_forward_parallel2(self):
        model = parallel_resnet18_with_fusion_gate(parallel=2)
        assert(isinstance(model(torch.rand(1, 3, 224, 224)), torch.Tensor))
