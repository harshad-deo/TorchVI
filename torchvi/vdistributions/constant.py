from collections.abc import Iterable
from typing import Iterable
import torch

from torchvi.vmodule import VModule


class Constant(VModule):
    def __init__(self, value):
        super().__init__()
        self.register_buffer('value', value)
        self.register_buffer('constraint_contrib', torch.squeeze(torch.zeros(1)))

    def forward(self, x):
        return self.value, self.constraint_contrib

    def sample(self, x, size):
        return self.value.repeat(size)


def wrap_if_constant(x):
    if isinstance(x, VModule):
        return x
    if isinstance(x, int) or isinstance(x, float):
        tensor = torch.tensor(x, dtype=torch.double, requires_grad=False)
    elif isinstance(x, Iterable):
        tensor = torch.tensor(x, requires_grad=False)
    elif isinstance(x, torch.Tensor):
        tensor = x
    else:
        raise TypeError(f'Unsupported type for wrapping. Expected int, float or tensor, got: {type(x)}')
    return Constant(tensor)
