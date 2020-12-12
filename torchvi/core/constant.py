from collections.abc import Iterable
import torch
from torch import nn
from typing import Iterable

from torchvi.core.ast import SingleNodeIdentity
from torchvi.core.constraint import Constraint
from torchvi.core.vmodule import VModule


class ConstantImpl(nn.Module):
    def __init__(self, value: torch.Tensor, name: str):
        super().__init__()
        self.name = name
        self.register_buffer('value', value)
        self.register_buffer('constraint_contrib', torch.squeeze(torch.zeros(1)))

    def forward(self):
        return self.value, Constraint.new(self.name, self.constraint_contrib)

    def sample(self, size) -> torch.Tensor:
        return self.value.repeat(size)

    def extra_repr(self) -> str:
        return f'name={self.name}, value={self.value}'


class Constant(VModule):
    def __init__(self, value: torch.Tensor, name: str):
        super().__init__(name=name)
        backing_name = f'{name}_backing'
        self._module_dict[backing_name] = ConstantImpl(value=value, name=backing_name)
        terminal_node = SingleNodeIdentity(name=self.name, arg=backing_name)
        self._terminal_node = terminal_node
        self._graph_dict[self.name] = terminal_node


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
