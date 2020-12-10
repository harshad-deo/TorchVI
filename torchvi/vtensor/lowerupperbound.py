import torch

from torchvi.vmodule import VModule
from torchvi.vtensor import utils
from torchvi.vtensor.backing import Backing
from torchvi.vtensor.constraint import Constraint


class LowerUpperBound(VModule):
    def __init__(self, size, lower_bound, upper_bound, name=None):
        super().__init__()
        self.backing = Backing(size, name)
        utils.check_numeric(lower_bound, 'lower_bound')
        utils.check_numeric(upper_bound, 'upper_bound')
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def forward(self, x):
        zeta, constraint_contrib = self.backing.forward()

        jac_contrib = zeta - 2 * (1 + (-zeta).exp()).log()
        constraint_contrib += Constraint.new(f'{self.backing.name}_lbub', jac_contrib.sum())

        return torch.sigmoid(zeta), constraint_contrib

    def sample(self, x, size):
        zeta = self.backing.sample(size)
        return torch.sigmoid(zeta)

    def extra_repr(self):
        return f'name={self.backing.name} size={self.backing.size}, lower_bound={self.lower_bound}, upper_bound={self.upper_bound}'
