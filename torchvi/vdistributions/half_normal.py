from torch import distributions

from torchvi.vmodule import VModule
from torchvi.vtensor.lowerbound import LowerBound
from torchvi.vtensor.constraint import Constraint
from torchvi.vdistributions.constant import wrap_if_constant


class HalfNormal(VModule):
    def __init__(self, size, scale, name=None):
        super().__init__()
        self.size = size
        self.node = LowerBound(size=size, lower_bound=0.0, name=name)
        self.scale = wrap_if_constant(scale, name=f'{self.node.backing.name}_scale')

    def forward(self, x):
        zeta, constraint_contrib = self.node.forward(x)

        scale, scale_constraint = self.scale.forward(x)
        constraint_contrib += scale_constraint

        prior = distributions.HalfNormal(scale)
        name = self.node.backing.name
        constraint_contrib += Constraint.new(f'{name}_prior', prior.log_prob(zeta).sum())

        return zeta, constraint_contrib

    def sample(self, x, size):
        return self.node.sample(x, size)

    def extra_repr(self):
        return f'size={self.size}, loc={self.loc}, scale={self.scale}'
