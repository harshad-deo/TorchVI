import torch

from torchvi.vmodule import VModule
from torchvi.vtensor import utils
from torchvi.vtensor.backing import Backing


class LowerBound(VModule):
    def __init__(self, size, lower_bound, name=None):
        super().__init__()
        self.backing = Backing(size, name)
        utils.check_numeric(lower_bound, 'lower_bound')
        self.lower_bound = lower_bound

    def forward(self, x):
        zeta, constraint_contrib = self.backing.forward()
        constraint_contrib += zeta.sum()

        return self.lower_bound + zeta.exp(), constraint_contrib

    def sample(self, x, size):
        zeta = self.backing.sample(size)
        return self.lower_bound + torch.exp(zeta)

    def extra_repr(self):
        return f'name={self.backing.name} size={self.backing.size}, lower_bound={self.lower_bound}'
