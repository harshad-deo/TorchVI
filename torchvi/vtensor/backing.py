import torch
from torch import nn

from torchvi.core.constraint import Constraint
from torchvi.core.vmodule import VModule
from torchvi.vtensor import utils


class Backing(VModule):
    def __init__(self, size, name: str):
        super().__init__(name=name)
        self.size = utils.to_size(size)
        self.mu = nn.Parameter(torch.Tensor(self.size), requires_grad=True)
        self.omega = nn.Parameter(torch.Tensor(self.size), requires_grad=True)

        self.mu.data.normal_()
        self.omega.data.normal_()

    def forward(self):
        device = self.mu.device

        eta = torch.randn(self.size, device=device)
        zeta = self.mu + eta * self.omega.exp()
        constraint_contrib = Constraint.new(self.name, self.omega.sum())
        return zeta, constraint_contrib

    def sample(self, size) -> torch.Tensor:
        device = self.mu.device

        sample_size = [size] + list(self.size)
        eta = torch.randn(sample_size, device=device)
        mu = self.mu.detach().unsqueeze(0)
        omega = self.omega.detach().unsqueeze(0)
        return mu + eta * omega.exp()

    def extra_repr(self) -> str:
        return f'name={self.name}, size={self.size}'
