import pytest
import torch

from binarizers import WeightBinarizer


class TestWeightBinarizer:

    def test_ones(self):
        binarizer = WeightBinarizer()
        assert(torch.equal(binarizer.forward(torch.ones((3, 4, 5, 6))), torch.ones((3, 4, 5, 6))))

    def test_rand(self):
        binarizer = WeightBinarizer()
        inp =torch.randn((3, 4, 5, 6))
        abs_mean = inp.abs().mean()
        assert(torch.equal(binarizer.forward(inp), inp.sign() * abs_mean))

    def test_backward_rand(self):
        binarizer = WeightBinarizer()
        input = torch.randn((3, 4, 5, 6), requires_grad=True)
        out = binarizer.forward(input)
        out.retain_grad()  # All the intermediate variables' gradients are removed during the backward() call.
                            # If you want to retain those gradients, call .retain_grad() before calling backward()
        out.mean().backward()
        assert(torch.equal(out.grad, input.grad))