import torch
from torch import distributions

from torchvi.vmodule import VModule
from torchvi.vtensor.lowerupperbound import LowerUpperBound
from torchvi.vdistributions.constant import wrap_if_constant


class Beta(VModule):
    def __init__(self, size, alpha, beta):
        super().__init__()
        self.size = size
        self.alpha = wrap_if_constant(alpha)
        self.beta = wrap_if_constant(beta)
        self.backing = LowerUpperBound(size, 0, 1)

    def forward(self, x):
        zeta, constraint_contrib = self.backing.forward(x)

        alpha, alpha_constraint = self.alpha.forward(x)
        beta, beta_constraint = self.beta.forward(x)
        constraint_contrib += alpha_constraint + beta_constraint

        prior = distributions.Beta(alpha, beta)
        constraint_contrib += prior.log_prob(zeta).sum()

        return zeta, constraint_contrib

    def sample(self, x, size, device):
        return self.backing.sample(x, size, device)

    def extra_repr(self):
        return f'size={self.size}, alpha={self.alpha}, beta={self.beta}'
