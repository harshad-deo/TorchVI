import torch
from torch import distributions

from torchvi.vmodule import VModule
from torchvi.vtensor.unconstrained import Unconstrained
from torchvi.vdistributions.constant import wrap_if_constant


class Normal(VModule):
    def __init__(self, size, loc, scale, name=None):
        super().__init__()
        self.size = size
        self.node = Unconstrained(size=size, name=name)
        self.loc = wrap_if_constant(loc, name=f'{self.node.backing.name}_loc')
        self.scale = wrap_if_constant(scale, name=f'{self.node.backing.name}_scale')

    def forward(self, x):
        zeta, constraint_contrib = self.node.forward(x)

        loc, loc_constraint = self.loc.forward(x)
        scale, scale_constraint = self.scale.forward(x)
        constraint_contrib += loc_constraint + scale_constraint

        prior = distributions.Normal(loc, scale)
        constraint_contrib += prior.log_prob(zeta).sum()

        return zeta, constraint_contrib

    def sample(self, x, size):
        return self.node.sample(x, size)

    def extra_repr(self):
        return f'size={self.size}, loc={self.loc}, scale={self.scale}'
