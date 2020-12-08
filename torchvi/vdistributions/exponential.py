from torch import distributions

from torchvi.vmodule import VModule
from torchvi.vdistributions.constant import wrap_if_constant
from torchvi.vtensor.lowerbound import LowerBound


class Exponential(VModule):
    def __init__(self, size, rate):
        super().__init__()
        self.size = size
        self.rate = wrap_if_constant(rate)
        self.node = LowerBound(size, 0.0)

    def forward(self, x):
        zeta, constraint_contrib = self.node.forward(x)

        rate, rate_constraint = self.rate.forward(x)
        constraint_contrib += rate_constraint

        prior = distributions.Exponential(rate)
        constraint_contrib += prior.log_prob(zeta).sum()

        return zeta, constraint_contrib

    def sample(self, x, size):
        return self.node.sample(x, size)

    def extra_repr(self):
        return f'size={self.size}, rate={self.rate}'
