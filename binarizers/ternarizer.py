import torch.nn as nn
from torch.autograd import Function


class TernarizerFunction(Function):  # from Trained Ternary Quantization
    @staticmethod
    def forward(ctx, input, pos_weight, neg_weight, threshold):
        ctx.input = input
        ctx.pos_weight = pos_weight
        ctx.neg_weigt = neg_weight
        ctx.threshold = threshold
        input[input.abs() < threshold] = 0
        input[input > threshold] = pos_weight
        input[input < -threshold] = neg_weight
        return input

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.input
        grad_pos = grad_output[x > ctx.threshold].sum()
        grad_neg = grad_output[x < -ctx.threshold].sum()
        grad_output[x > ctx.threshold] *= ctx.pos_weight
        grad_output[x < -ctx.threshold] *= ctx.neg_weight
        return grad_output, grad_pos, grad_neg, None


class Ternarizer(nn.Module):
    def __init__(self, threshold):
        super(Ternarizer, self).__init__()
        assert threshold > 0
        self.threshold = threshold
        self.pos_weight = nn.Parameter(torch.Tensor([1]))
        self.neg_weight = nn.Parameter(torch.Tensor([1]))

    def forward(self, input):
        return TernarizerFunction.apply(input, self.pos_weight, self.neg_weight, self.threshold)

    def extra_repr(self):
        return 'positive weight={:.3f}, negative weight={:.3f}, threshold={:.3f}'.format(
            self.pos_weight, self.neg_weight, self.threshold)
