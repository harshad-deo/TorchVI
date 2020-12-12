import torch
from torch import nn

from torchvi.core.ast import SingleNodeIdentity
from torchvi.core.constraint import Constraint
from torchvi.core.vmodule import VModule
from torchvi.vtensor import utils
from torchvi.vtensor.unconstrained import UnconstrainedImpl


class LowerBoundImpl(nn.Module):
    def __init__(self, size, lower_bound, name: str):
        super().__init__()
        self.name = name
        self.backing = UnconstrainedImpl(size=size, name=f'{self.name}_backing')
        utils.check_numeric(lower_bound, 'lower_bound')
        self.lower_bound = lower_bound

    def forward(self):
        zeta, constraint_contrib = self.backing.forward()
        constraint_contrib += Constraint.new(f'{self.backing.name}_lb', zeta.sum())

        return self.lower_bound + zeta.exp(), constraint_contrib

    def sample(self, size) -> torch.Tensor:
        zeta = self.backing.sample(size)
        return self.lower_bound + torch.exp(zeta)

    def extra_repr(self) -> str:
        return f'name={self.name}, lower_bound={self.lower_bound}'


class LowerBound(VModule):
    def __init__(self, size, lower_bound, name: str):
        super().__init__(name)
        impl_name = f'{self.name}_impl'
        self._module_dict[impl_name] = LowerBoundImpl(size=size, lower_bound=lower_bound, name=impl_name)
        terminal_node = SingleNodeIdentity(name=self.name, arg=impl_name)
        self._terminal_node = terminal_node
        self._graph_dict[self.name] = terminal_node
