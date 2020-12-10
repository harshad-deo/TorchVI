from collections.abc import Iterable
from typing import Iterable
import torch

from torchvi.vmodule import VModule
from torchvi.vtensor.constraint import Constraint


class Constant(VModule):
    def __init__(self, value, name):
        super().__init__()
        self.__name = name
        self.register_buffer('value', value)
        self.register_buffer('constraint_contrib', torch.squeeze(torch.zeros(1)))

    def forward(self, x):
        return self.value, Constraint.new(self.name, self.constraint_contrib)

    def sample(self, x, size):
        return self.value.repeat(size)

    def extra_repr(self):
        return f'name={self.__name} size={self.size}, alpha={self.alpha}, beta={self.beta}'

    @property
    def name(self):
        return self.__name


def wrap_if_constant(x, name: str):
    if isinstance(x, VModule):
        return x
    if isinstance(x, int) or isinstance(x, float):
        tensor = torch.tensor(x, requires_grad=False)
    elif isinstance(x, Iterable):
        tensor = torch.tensor(x, requires_grad=False)
    elif isinstance(x, torch.Tensor):
        tensor = x
    else:
        raise TypeError(f'Unsupported type for wrapping. Expected int, float or tensor, got: {type(x)}')
    return Constant(value=tensor, name=name)
