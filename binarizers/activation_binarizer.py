from torch.autograd import Function
import torch.nn as nn


class ActivationBinarizerFunction(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.input.clamp(-1, 1)
        return (2 - 2 * x * x.sign()) * grad_output


class ActivationBinarizer(nn.Module):
    def forward(self, input):
        return ActivationBinarizerFunction.apply(input)
