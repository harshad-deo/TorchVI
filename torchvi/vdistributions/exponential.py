from torch import distributions

from torchvi.core.vmodule import VModule
from torchvi.vdistributions.constant import wrap_if_constant
from torchvi.vtensor.lowerbound import LowerBound
from torchvi.vtensor.constraint import Constraint


class Exponential(VModule):
    def __init__(self, size, rate, name=None):
        super().__init__()
        self.size = size
        self.node = LowerBound(size=size, lower_bound=0.0, name=name)
        self.rate = wrap_if_constant(rate, name=f'{self.node.backing.name}_rate')

    def forward(self, x):
        zeta, constraint_contrib = self.node.forward(x)

        rate, rate_constraint = self.rate.forward(x)
        constraint_contrib += rate_constraint

        name = self.node.backing.name
        prior = distributions.Exponential(rate)
        constraint_contrib += Constraint.new(f'{name}_prior', prior.log_prob(zeta).sum())

        return zeta, constraint_contrib

    def sample(self, x, size):
        return self.node.sample(x, size)

    def extra_repr(self):
        return f'size={self.size}, rate={self.rate}'
