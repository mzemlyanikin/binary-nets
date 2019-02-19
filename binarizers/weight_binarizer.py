import torch.nn as nn
from torch.autograd import Function


class WeightBinarizerFunction(Function):
    @staticmethod
    def forward(ctx, x):
        # ctx is a context object that can be used to stash information
        # for backward computation
        return x.sign() * x.abs().mean()

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        return grad_output


class WeightBinarizer(nn.Module):
    def forward(self, input):
        return WeightBinarizerFunction.apply(input)
