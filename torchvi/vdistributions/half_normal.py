import torch
from torch import distributions

from torchvi.core.vmodule import VModule
from torchvi.vtensor.lowerbound import LowerBound
from torchvi.vtensor.constraint import Constraint
from torchvi.vdistributions.constant import wrap_if_constant


class HalfNormal(VModule):
    def __init__(self, size, scale, name: str = None):
        super().__init__(name=name)
        self.node = LowerBound(size=size, lower_bound=0.0, name=f'{self.name}_node')
        self.scale = wrap_if_constant(scale, name=f'{self.name}_scale')

    def forward(self, x):
        zeta, constraint_contrib = self.node.forward(x)

        scale, scale_constraint = self.scale.forward(x)
        constraint_contrib += scale_constraint

        prior = distributions.HalfNormal(scale)
        constraint_contrib += Constraint.new(f'{self.name}_prior', prior.log_prob(zeta).sum())

        return zeta, constraint_contrib

    def sample(self, x, size) -> torch.Tensor:
        return self.node.sample(x, size)

    def extra_repr(self) -> str:
        return f'name={self.name}'
