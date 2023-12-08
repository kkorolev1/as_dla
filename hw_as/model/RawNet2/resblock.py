import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class FMS(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.linear = nn.Linear(num_channels, num_channels)
        self.act = nn.Sigmoid()
        
    def forward(self, x):
        scales = x.mean(dim=-1)
        scales = self.act(self.linear(scales))
        return x * scales.unsqueeze(-1)


class ResLayer(nn.Module):
    def __init__(self, in_channels, out_channels, negative_slope=0.01):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.LeakyReLU(negative_slope),
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(negative_slope),
            nn.Conv1d(out_channels, out_channels, 3, padding=1)
        )
        self.reproj = nn.Conv1d(in_channels, out_channels, 1)
        self.tail = nn.Sequential(
            nn.MaxPool1d(3),
            FMS(out_channels)
        )
        
    def forward(self, x):
        x = self.sequential(x) + self.reproj(x)
        x = self.tail(x)
        return x
    
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, negative_slope=0.01):
        super().__init__()
        assert num_layers >= 1
        self.first = ResLayer(in_channels, out_channels, negative_slope)
        self.tail = nn.ModuleList([
            ResLayer(out_channels, out_channels, negative_slope)
            for _ in range(num_layers - 1)
        ])
        
    def forward(self, x):
        x = self.first(x)
        for layer in self.tail:
            x = layer(x)
        return x