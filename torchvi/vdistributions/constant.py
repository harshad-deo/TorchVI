from collections.abc import Iterable
import torch
from typing import Iterable

from torchvi.core.vmodule import VModule
from torchvi.vtensor.constraint import Constraint


class Constant(VModule):
    def __init__(self, value: torch.Tensor, name: str):
        super().__init__(name=name)
        self.register_buffer('value', value)
        self.register_buffer('constraint_contrib', torch.squeeze(torch.zeros(1)))

    def forward(self, x):
        return self.value, Constraint.new(self.name, self.constraint_contrib)

    def sample(self, x, size) -> torch.Tensor:
        return self.value.repeat(size)

    def extra_repr(self) -> str:
        return f'name={self.name}, value={self.value}'


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
