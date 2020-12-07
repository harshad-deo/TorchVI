import torch
from torch import distributions

from torchvi.vmodule import VModule
from torchvi.vtensor.unconstrained import Unconstrained
from torchvi.vdistributions.constant import wrap_if_constant


class Normal(VModule):
    def __init__(self, size, loc, scale):
        super().__init__()
        self.size = size
        self.loc = wrap_if_constant(loc)
        self.scale = wrap_if_constant(scale)
        self.backing = Unconstrained(size)

    def forward(self, x, device):
        zeta, constraint_contrib = self.backing.forward(x, device)

        loc, loc_constraint = self.loc.forward(x, device)
        scale, scale_constraint = self.scale.forward(x, device)
        constraint_contrib += loc_constraint + scale_constraint

        prior = distributions.Normal(loc, scale)
        constraint_contrib += prior.log_prob(zeta).sum()

        return zeta, constraint_contrib

    def sample(self, x, size, device):
        return self.backing.sample(x, size, device)

    def extra_repr(self):
        return f'size={self.size}, loc={self.loc}, scale={self.scale}'
