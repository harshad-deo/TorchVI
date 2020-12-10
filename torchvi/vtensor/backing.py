import torch
from torch import nn
import uuid

from torchvi.vmodule import VModule
from torchvi.vtensor import utils


class Backing(VModule):
    def __init__(self, size, name=None):
        super().__init__()
        if name is None:
            name = uuid.uuid4().hex
        self.__name = name
        self.size = utils.to_size(size)
        self.mu = nn.Parameter(torch.Tensor(self.size), requires_grad=True)
        self.omega = nn.Parameter(torch.Tensor(self.size), requires_grad=True)

        self.mu.data.normal_()
        self.omega.data.normal_()

    def forward(self):
        device = self.mu.device

        eta = torch.randn(self.size, device=device)
        zeta = self.mu + eta * self.omega.exp()
        constraint_contrib = torch.squeeze(self.omega.sum())
        return zeta, constraint_contrib

    def sample(self, size):
        device = self.mu.device

        sample_size = [size] + list(self.size)
        eta = torch.randn(sample_size, device=device)
        mu = self.mu.detach().unsqueeze(0)
        omega = self.omega.detach().unsqueeze(0)
        return mu + eta * omega.exp()

    @property
    def name(self):
        return self.__name

    def extra_repr(self):
        return f'name={self.__name} size={self.size}'
