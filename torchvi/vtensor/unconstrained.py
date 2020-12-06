import torch
from torch import nn

from torchvi.vmodule import VModule
from torchvi.vtensor import utils


class Unconstrained(VModule):
    def __init__(self, size):
        super().__init__()
        self.size = utils.to_size(size)

        self.mu = nn.Parameter(torch.Tensor(self.size), requires_grad=True)
        self.omega = nn.Parameter(torch.Tensor(self.size), requires_grad=True)

        self.mu.data.normal_()
        self.omega.data.normal_()

        self.register_buffer('degen', torch.zeros(1))

    def forward(self, device):
        eta = torch.randn(self.size, device=device)
        y = self.mu + eta * self.omega.exp()
        entropy_contrib = self.omega.sum()

        return y, self.degen, self.degen, entropy_contrib

    def sample(self, size, device):
        sample_size = [size] + self.size
        eta = torch.randn(sample_size, device=device)
        mu = self.mu.detach().unsqueeze(0)
        omega = self.omega.detach().unsqueeze(0)
        return mu + eta * omega.exp()

    def extra_repr(self):
        return f'size={self.size}'
