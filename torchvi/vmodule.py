from abc import ABC, abstractmethod
from torch import nn


class VModule(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def sample(self, x, size, device):
        pass

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
