import torch
from torch import distributions, nn

from torchvi.vmodule import VModule
from torchvi.vtensor.lowerupperbound import LowerUpperBound
from torchvi.vdistributions.deterministic import wrap_if_deterministic


class Beta(VModule):
    def __init__(self, size, alpha, beta):
        super().__init__()
        self.alpha = wrap_if_deterministic(alpha)
        self.beta = wrap_if_deterministic(beta)
        self.backing = LowerUpperBound(size, 0, 1)

    def forward(self, x, device):
        zeta, constraint_contrib = self.backing.forward(x, device)

        alpha, alpha_constraint = self.alpha.forward(x, device)
        beta, beta_constraint = self.beta.forward(x, device)
        constraint_contrib += alpha_constraint + beta_constraint

        prior = distributions.Beta(alpha, beta)
        constraint_contrib += torch.squeeze(prior.log_prob(zeta))

        return zeta, constraint_contrib

    def sample(self, x, size, device):
        return self.backing.sample(x, size, device)
