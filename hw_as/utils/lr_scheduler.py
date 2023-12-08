import torch
import math

from torch.optim.lr_scheduler import OneCycleLR


class ReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def get_last_lr(self):
        return [self.optimizer.param_groups[0]['lr']]


class TransformerLrScheduler:
    def __init__(self, optimizer, d_model, warmup_steps, multiplier=5):
        self._optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.n_steps = 0
        self.multiplier = multiplier
        self._lr = 0

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
    
    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict) 

    def step(self):
        self.n_steps += 1
        self._lr = self._get_lr()
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = self._lr

    def _get_lr(self):
        return self.multiplier * (self.d_model ** -0.5) * min(self.n_steps ** (-0.5), self.n_steps * (self.warmup_steps ** (-1.5)))
  
    def get_last_lr(self):
        return [self._lr]


class CosineAnnealingWithWarmupLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup_steps: int, max_steps: int):
        self.warmup = warmup_steps
        self.max_steps = max_steps
        super().__init__(optimizer)

    def get_lr(self):
        return self.get_last_lr()

    def get_last_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + math.cos(math.pi * epoch / self.max_steps))
        lr_factor *= min(epoch / self.warmup, 1.0)
        return lr_factor