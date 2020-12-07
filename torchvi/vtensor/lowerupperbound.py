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

    def forward(self, device):
        zeta, constraint_contrib = super().forward(device)

        jac_contrib = zeta - 2 * (1 + (-zeta).exp()).log()
        constraint_contrib += torch.squeeze(jac_contrib)

        return torch.sigmoid(zeta), constraint_contrib

    def sample(self, size, device):
        zeta = super().sample(size, device)
        return torch.sigmoid(zeta)
