from torch._C import dtype
from torchvi.core.ast import SingleNodeIdentity
import torch
from torch import nn
import torch.nn.functional as F

from torchvi.core.constraint import Constraint
from torchvi.core.vmodule import VModule
from torchvi.vtensor.backing import Backing


class SimplexImpl(nn.Module):
    def __init__(self, size: int, name: str):
        super().__init__()

        if not isinstance(size, int):
            raise TypeError(f'Simplex size must be an integer. Got: {type(size)}')
        if size <= 1:
            raise ValueError(f'Simplex size must be greater than 1. Got: {size}')

        self.name = name
        self.size = size
        self.backing = Backing(size - 1, f'{self.name}_backing')
        ar = torch.arange(1, size, dtype=torch.float64, requires_grad=False)
        grad_scale = size - ar
        self.register_buffer('grad_scale', grad_scale)
        log_add = -grad_scale.log()
        self.register_buffer('log_add', log_add)

    def __simplex_transform(self, zeta):
        z = torch.sigmoid(zeta + self.log_add)
        zl = F.pad(input=z, pad=[1, 0], value=0)  # left pad
        zr = F.pad(input=z, pad=[0, 1], value=1)  # right pad
        theta = (1 - zl).cumprod(-1) * zr
        return z, theta

    def forward(self):
        zeta, constraint_contrib = self.backing.forward()
        z, theta = self.__simplex_transform(zeta)

        simplex_constraint_contrib = z.log() + self.grad_scale * (1 - z).log()

        constraint_contrib += Constraint.new(f'{self.backing.name}_simplex', simplex_constraint_contrib.sum())
        return theta, constraint_contrib

    def sample(self, size) -> torch.Tensor:
        zeta = self.backing.sample(size)
        _, theta = self.__simplex_transform(zeta)
        return theta

    def extra_repr(self) -> str:
        return f'name={self.name}, size={self.size}'


class Simplex(VModule):
    def __init__(self, size: int, name: str):
        super().__init__(name)
        impl_name = f'{self.name}_impl'
        self._module_dict[impl_name] = SimplexImpl(size=size, name=impl_name)
        terminal_node = SingleNodeIdentity(name=self.name, arg=impl_name)
        self._terminal_node = terminal_node
        self._graph_dict[self.name] = terminal_node
