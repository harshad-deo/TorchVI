from abc import ABC, abstractmethod
import torch
from torch import nn
import uuid


class VModule(nn.Module, ABC):
    def __init__(self, name: str = None):
        super().__init__()
        if name is None:
            name = uuid.uuid4().hex
        self.__name = name

    @abstractmethod
    def sample(self, x, size) -> torch.Tensor:
        pass

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def name(self) -> str:
        return self.__name
