import torch
from torch import distributions

from torchvi.core.vmodule import VModule
from torchvi.vtensor.constraint import Constraint
from torchvi.vtensor.unconstrained import Unconstrained
from torchvi.vdistributions.constant import wrap_if_constant


class Laplace(VModule):
    def __init__(self, size, loc, scale, name: str = None):
        super().__init__(name)
        self.node = Unconstrained(size=size, name=f'{self.name}_node')
        self.loc = wrap_if_constant(loc, name=f'{self.name}_loc')
        self.scale = wrap_if_constant(scale, name=f'{self.name}_scale')

    def forward(self, x):
        zeta, constraint_contrib = self.node.forward(x)

        loc, loc_constraint = self.loc.forward(x)
        scale, scale_constraint = self.scale.forward(x)

        prior = distributions.Laplace(loc=loc, scale=scale)
        constraint_contrib += Constraint.new(f'{self.name}_prior',
                                             prior.log_prob(zeta).sum()) + loc_constraint + scale_constraint

        return zeta, constraint_contrib

    def sample(self, x, size) -> torch.Tensor:
        return self.node.sample(x, size)

    def extra_repr(self) -> str:
        return f'name={self.name}'
