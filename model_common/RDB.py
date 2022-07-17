'''
    Residual Dense Block called RDB
'''

import torch
from torch import nn


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size//2), bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers, kernel_size=5):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate, kernel_size=kernel_size) for i in range(num_layers)])

        # local feature fusion
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)  #k=1

    def forward(self, x, lrl=True):
        if lrl:
            return x + self.lff(self.layers(x))  # local residual learning
        else:
            return self.layers(x)