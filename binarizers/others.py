import torch.nn as nn


class Identity(nn.Module):
    def forward(self, x):
        return x

class Sign(nn.Module):
    def forward(self, input):
        x = input.clone()
        x[x >= 0] = 1
        x[x < 0] = -1
        return x