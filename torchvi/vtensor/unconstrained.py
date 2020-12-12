import torch
from torch import nn

from torchvi.core.ast import SingleNodeIdentity
from torchvi.core.constraint import Constraint
from torchvi.core.vmodule import VModule
from torchvi.vtensor import utils


class UnconstrainedImpl(nn.Module):
    def __init__(self, size, name: str):
        super().__init__()
        self.__name = name
        self.size = utils.to_size(size)
        self.mu = nn.Parameter(torch.Tensor(self.size), requires_grad=True)
        self.omega = nn.Parameter(torch.Tensor(self.size), requires_grad=True)

        self.mu.data.normal_()
        self.omega.data.normal_()

    def forward(self):
        device = self.mu.device

        eta = torch.randn(self.size, device=device)
        zeta = self.mu + eta * self.omega.exp()
        constraint_contrib = Constraint.new(self.name, self.omega.sum())
        return zeta, constraint_contrib

    def sample(self, size) -> torch.Tensor:
        device = self.mu.device

        sample_size = [size] + list(self.size)
        eta = torch.randn(sample_size, device=device)
        mu = self.mu.detach().unsqueeze(0)
        omega = self.omega.detach().unsqueeze(0)
        return mu + eta * omega.exp()

    @property
    def name(self) -> str:
        return self.__name

    def extra_repr(self) -> str:
        return f'name={self.name} size={self.size}'


class Unconstrained(VModule):
    def __init__(self, size, name: str):
        super().__init__(name=name)
        backing_name = f'{self.name}_backing'
        self._module_dict[backing_name] = UnconstrainedImpl(size=size, name=backing_name)
        terminal_node = SingleNodeIdentity(name=self.name, arg=backing_name)
        self._terminal_node = terminal_node
        self._graph_dict[self.name] = terminal_node
