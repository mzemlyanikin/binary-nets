import pytest
import torch

from binarizers import ActivationBinarizer


class TestActivationBinarizer:

    def test_ones(self):
        binarizer = ActivationBinarizer()
        assert(torch.equal(binarizer.forward(torch.ones((3, 4, 5, 6))), torch.ones((3, 4, 5, 6))))

    def test_rand(self):
        binarizer = ActivationBinarizer()
        inp = torch.randn((3, 4, 5, 6))
        assert(torch.equal(binarizer.forward(inp), inp.sign()))

    def test_backward_rand(self):
        binarizer = ActivationBinarizer()
        input = torch.randn((3, 4, 5, 6), requires_grad=True)
        out = binarizer.forward(input)
        out.retain_grad()  # All the intermediate variables' gradients are removed during the backward() call.
                            # If you want to retain those gradients, call .retain_grad() before calling backward()
        out.mean().backward()
        db_dx = 2 - 2 * input.sign() * input.clamp(-1, 1)
        assert(torch.equal(db_dx * out.grad, input.grad))