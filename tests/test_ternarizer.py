import random
import pytest
import torch

from binarizers import Ternarizer


class TestTernarizer:

    def test_ones_thresh_less_one(self):
        binarizer = Ternarizer(threshold=0.5)
        assert(torch.equal(binarizer.forward(torch.ones((3, 4, 5, 6))), torch.ones((3, 4, 5, 6))))

    def test_ones_thresh_more_one(self):
        binarizer = Ternarizer(threshold=1.5)
        assert (torch.equal(binarizer.forward(torch.ones((3, 4, 5, 6))), torch.zeros((3, 4, 5, 6))))

    def test_forward_rand(self):
        rand_thresh = random.random()
        binarizer = Ternarizer(threshold=rand_thresh)
        input = torch.randn((3, 4, 5, 6))
        out = input
        out[input.abs() <= rand_thresh] = 0
        out[input < -rand_thresh] = -1  # initial parameters are 1 and -1
        out[input > rand_thresh] = 1
        assert (torch.equal(binarizer.forward(input), out))

    def test_backward_rand(self):
        binarizer = Ternarizer(threshold=random.random())
        input = torch.randn((3, 4, 5, 6), requires_grad=True)
        out = binarizer.forward(input)
        out.retain_grad()  # All the intermediate variables' gradients are removed during the backward() call.
                            # If you want to retain those gradients, call .retain_grad() before calling backward()
        out.mean().backward()
        print(binarizer.pos_weight.grad)
        assert(torch.equal(out.grad, input.grad))

    def test_backward_rand_pos_weight(self):
        rand_thresh = random.random()
        binarizer = Ternarizer(threshold=rand_thresh)
        input = torch.randn((3, 4, 5, 6), requires_grad=True)
        out = binarizer.forward(input)
        out.retain_grad()  # All the intermediate variables' gradients are removed during the backward() call.
                            # If you want to retain those gradients, call .retain_grad() before calling backward()
        out.mean().backward()
        assert(torch.equal(out.grad[input > rand_thresh].sum(), binarizer.pos_weight.grad))

    def test_backward_rand_neg_weight(self):
        rand_thresh = random.random()
        binarizer = Ternarizer(threshold=rand_thresh)
        input = torch.randn((3, 4, 5, 6), requires_grad=True)
        out = binarizer.forward(input)
        out.retain_grad()  # All the intermediate variables' gradients are removed during the backward() call.
                            # If you want to retain those gradients, call .retain_grad() before calling backward()
        out.mean().backward()
        assert(torch.equal(out.grad[input < -rand_thresh].sum(), binarizer.neg_weight.grad))