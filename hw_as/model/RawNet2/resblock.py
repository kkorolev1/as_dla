import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class FMS(nn.Module):
    def __init__(self, num_channels, add=True, mul=True):
        super().__init__()
        self.linear = nn.Linear(num_channels, num_channels)
        self.act = nn.Sigmoid()
        self.add = add
        self.mul = mul
        
    def forward(self, x):
        scales = F.adaptive_avg_pool1d(x, 1).reshape(x.shape[0], -1)
        scales = self.act(self.linear(scales)).reshape(x.shape[0], x.shape[1], -1)
        if self.mul:
            x = x * scales
        if self.add:
            x = x + scales
        return x


class ResLayer(nn.Module):
    def __init__(self, in_channels, out_channels, negative_slope=0.3, is_first=False):
        super().__init__()
        if not is_first:
            self.head = nn.Sequential(
                nn.BatchNorm1d(in_channels),
                nn.LeakyReLU(negative_slope)
            )
        else:
            self.head = nn.Identity()
        self.sequential = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(negative_slope),
            nn.Conv1d(out_channels, out_channels, 3, padding=1)
        )
        if in_channels != out_channels:
            self.reproj = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.reproj = nn.Identity()
        self.tail = nn.Sequential(
            nn.MaxPool1d(3),
            FMS(out_channels)
        )
        
    def forward(self, x):
        x = self.head(x)
        x = self.sequential(x) + self.reproj(x)
        x = self.tail(x)
        return x
    
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, sep_first=False, negative_slope=0.3):
        super().__init__()
        assert num_layers >= 1
        self.first = ResLayer(in_channels, out_channels, negative_slope, is_first=sep_first)
        self.tail = nn.ModuleList([
            ResLayer(out_channels, out_channels, negative_slope)
            for _ in range(num_layers - 1)
        ])
        
    def forward(self, x):
        x = self.first(x)
        for layer in self.tail:
            x = layer(x)
        return x