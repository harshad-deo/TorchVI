import torch
from torch import distributions

from torchvi.core.vmodule import VModule
from torchvi.vdistributions.constant import wrap_if_constant
from torchvi.vtensor.lowerbound import LowerBound
from torchvi.vtensor.constraint import Constraint


class Exponential(VModule):
    def __init__(self, size, rate, name=None):
        super().__init__(name=name)
        self.node = LowerBound(size=size, lower_bound=0.0, name=f'{self.name}_node')
        self.rate = wrap_if_constant(rate, name=f'{self.name}_rate')

    def forward(self, x):
        zeta, constraint_contrib = self.node.forward(x)

        rate, rate_constraint = self.rate.forward(x)
        constraint_contrib += rate_constraint

        prior = distributions.Exponential(rate)
        constraint_contrib += Constraint.new(f'{self.name}_prior', prior.log_prob(zeta).sum())

        return zeta, constraint_contrib

    def sample(self, x, size) -> torch.tensor:
        return self.node.sample(x, size)

    def extra_repr(self) -> str:
        return f'name={self.name}'
