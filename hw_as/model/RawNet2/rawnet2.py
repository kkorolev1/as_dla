import torch
import torch.nn as nn

from hw_as.base import BaseModel
from hw_as.model.RawNet2.sinc import SincFilter
from hw_as.model.RawNet2.resblock import ResBlock

class RawNet2(BaseModel):
    def __init__(self, **config):
        super().__init__()
        self.sinc_filters = SincFilter(**config["sinc_config"])
        self.res1 = ResBlock(**config["res1_config"])
        self.res2 = ResBlock(**config["res2_config"])
        self.pre_gru = nn.Sequential(
            nn.BatchNorm1d(config["res2_config"]["out_channels"]),
            nn.LeakyReLU()
        )
        self.gru = nn.GRU(**config["gru_config"])
        self.linear = nn.Linear(
            config["gru_config"]["hidden_size"],
            config["gru_config"]["hidden_size"]
        )
        self.head = nn.Linear(config["gru_config"]["hidden_size"], 2)
        
    def forward(self, x):
        x = self.sinc_filters(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.pre_gru(x)
        x = self.gru(x.transpose(1, 2))[0][:, -1, :]
        x = self.linear(x)
        x = self.head(x)
        return x
        