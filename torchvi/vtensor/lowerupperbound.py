import torch

from torchvi.vtensor.unconstrained import Unconstrained
from torchvi.vtensor import utils


class LowerUpperBound(Unconstrained):
    def __init__(self, size, lower_bound, upper_bound):
        super().__init__(size)
        utils.check_numeric(lower_bound, 'lower_bound')
        utils.check_numeric(upper_bound, 'upper_bound')
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def forward(self, x):
        zeta, constraint_contrib = super().forward(x)

        jac_contrib = zeta - 2 * (1 + (-zeta).exp()).log()
        constraint_contrib += jac_contrib.sum()

        return torch.sigmoid(zeta), constraint_contrib

    def sample(self, x, size):
        zeta = super().sample(x, size)
        return torch.sigmoid(zeta)

    def extra_repr(self):
        return f'size={self.size}, lower_bound={self.lower_bound}, upper_bound={self.upper_bound}'
