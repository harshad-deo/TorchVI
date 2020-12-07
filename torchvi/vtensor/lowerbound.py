import torch

from torchvi.vtensor.unconstrained import Unconstrained
from torchvi.vtensor import utils


class LowerBound(Unconstrained):
    def __init__(self, size, lower_bound):
        super().__init__(size)
        utils.check_numeric(lower_bound, 'lower_bound')
        self.lower_bound = lower_bound

    def forward(self, x):
        zeta, constraint_contrib = super().forward(x)
        constraint_contrib += zeta.sum()

        return self.lower_bound + zeta.exp(), constraint_contrib

    def sample(self, x, size):
        zeta = super().sample(x, size)
        return self.lower_bound + torch.exp(zeta)

    def extra_repr(self):
        return f'size={self.size}, lower_bound={self.lower_bound}'
