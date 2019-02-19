import torch
import torch.nn as nn
from torch.autograd import Function


class TernarizerFunction(Function):  # from Trained Ternary Quantization
    @staticmethod
    def forward(ctx, input, pos_value, neg_value, threshold):
        ctx.save_for_backward(input, pos_value, neg_value)
        ctx.threshold = threshold
        input[input.abs() < threshold] = 0
        input[input > threshold] = pos_value
        input[input < -threshold] = -neg_value
        return input

    @staticmethod
    def backward(ctx, grad_output):
        x, pos_value, neg_value = ctx.saved_tensors
        grad_pos = grad_output[x > ctx.threshold].sum()
        grad_neg = grad_output[x < -ctx.threshold].sum()
        grad_output[x > ctx.threshold] *= pos_value
        grad_output[x < -ctx.threshold] *= neg_value
        return grad_output, grad_pos, grad_neg, None


class Ternarizer(nn.Module):
    def __init__(self, threshold=None):
        super(Ternarizer, self).__init__()
        self.threshold = threshold
        self.pos_weight = nn.Parameter(torch.Tensor([1]))
        self.neg_weight = nn.Parameter(torch.Tensor([1]))

    def forward(self, input, threshold=None):
        if threshold is None:
            if self.threshold is not None:
                threshold = self.threshold
            else:
                raise TypeError('Threshold parameter is required')
        return TernarizerFunction.apply(input, self.pos_weight, self.neg_weight, threshold)

    def extra_repr(self):
        return 'positive weight={:.3f}, negative weight={:.3f}, threshold={:.3f}'.format(
            self.pos_weight, self.neg_weight, self.threshold)
