import torch
from torch import distributions, nn

from torchvi.vmodule import VModule
from torchvi.vtensor.lowerupperbound import LowerUpperBound


class Beta(VModule):
    def __init__(self, size, alpha, beta):
        super().__init__()
        self.register_buffer('alpha', torch.tensor(alpha))
        self.register_buffer('beta', torch.tensor(beta))
        self.backing = LowerUpperBound(size, 0, 1)

    def forward(self, device):
        zeta, constraint_contrib = self.backing.forward(device)
        prior = distributions.Beta(self.alpha, self.beta)
        constraint_contrib += torch.squeeze(prior.log_prob(zeta))
        return zeta, constraint_contrib

    def sample(self, size, device):
        return self.backing.sample(size, device)
