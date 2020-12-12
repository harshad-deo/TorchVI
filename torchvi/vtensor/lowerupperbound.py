import torch
from torch import nn

from torchvi.core.ast import SingleNodeIdentity
from torchvi.core.constraint import Constraint
from torchvi.core.vmodule import VModule
from torchvi.vtensor import utils
from torchvi.vtensor.backing import Backing


class LowerUpperBoundImpl(nn.Module):
    def __init__(self, size, lower_bound, upper_bound, name: str):
        super().__init__()
        self.name = name
        self.backing = Backing(size, f'{self.name}_backing')
        lower_bound = utils.to_numeric_tensor(lower_bound, 'lower_bound')
        upper_bound = utils.to_numeric_tensor(upper_bound, 'upper_bound')
        self.register_buffer('lower_bound', lower_bound)
        self.register_buffer('range', upper_bound - lower_bound)

    def forward(self):
        zeta, constraint_contrib = self.backing()

        sigmoid_zeta = torch.sigmoid(zeta)
        res = self.lower_bound + self.range * sigmoid_zeta

        jac_contrib = 2 * sigmoid_zeta.log() - zeta  # the constant term is dropped because it wont affect the jac
        constraint_contrib += Constraint.new(f'{self.name}_lbub', jac_contrib.sum())

        return res, constraint_contrib

    def sample(self, size) -> torch.Tensor:
        zeta = self.backing.sample(size)
        return self.lower_bound.unsqueeze(0) + self.range.unsqueeze(0) * torch.sigmoid(zeta)

    def extra_repr(self) -> str:
        return f'name={self.name}, lower_bound={self.lower_bound}, upper_bound={self.range + self.lower_bound}'


class LowerUpperBound(VModule):
    def __init__(self, size, lower_bound, upper_bound, name: str):
        super().__init__(name)
        impl_name = f'{self.name}_impl'
        self._module_dict[impl_name] = LowerUpperBoundImpl(size=size,
                                                           lower_bound=lower_bound,
                                                           upper_bound=upper_bound,
                                                           name=impl_name)
        terminal_node = SingleNodeIdentity(name=self.name, arg=impl_name)
        self._terminal_node = terminal_node
        self._graph_dict[self.name] = terminal_node
