import pytest
import torch

from resnet import ResNet
from models import parallel_resnet18_with_squeeze


class TestParallelResnet18WithFusionGate:

    def test_dummy(self):
        assert(isinstance(parallel_resnet18_with_squeeze(), ResNet))

    def test_dummy_parallel2(self):
        assert(isinstance(parallel_resnet18_with_squeeze(parallel=2), ResNet))
    
    def test_forward(self):
        model = parallel_resnet18_with_squeeze()
        assert(isinstance(model(torch.rand(1, 3, 224, 224)), torch.Tensor))

    def test_forward_parallel2(self):
        model = parallel_resnet18_with_squeeze(parallel=2)
        assert(isinstance(model(torch.rand(1, 3, 224, 224)), torch.Tensor))
