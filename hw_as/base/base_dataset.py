import logging
import random
from typing import List
import os

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from hw_as.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, ind):
        return None

    def __len__(self):
        return len(self._index)
