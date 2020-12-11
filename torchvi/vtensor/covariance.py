import torch
from torch import nn

from torchvi.core.constraint import Constraint
from torchvi.core.vmodule import VModule


class Covariance(VModule):
    def __init__(self, size: int, name=None):
        super().__init__(name=name)
        if not isinstance(size, int):
            raise TypeError(f'size must be an integer. Got: {type(size)}')
        if size <= 1:
            raise ValueError(f'size must be greater than 1. Got: {size}')

        self.size = size
        self.tril_size = (size * (size - 1)) // 2

        diag_idx = torch.arange(size)
        tril_idx = torch.tril_indices(row=size, col=size, offset=-1)
        self.register_buffer('diag_idx', diag_idx)
        self.register_idx('tril_idx', tril_idx)

        diag_scale = size - torch.arange(size) + 1
        self.register_buffer('diag_scale', diag_scale)

        self.diag_mu = nn.Parameter(torch.Tensor(self.size), requires_grad=True)
        self.diag_omega = nn.Parameter(torch.Tensor(self.size), requires_grad=True)

        self.tril_mu = nn.Parameter(torch.Tensor(self.tril_size), requires_grad=True)
        self.tril_omega = nn.Parameter(torch.Tensor(self.tril_size), requires_grad=True)

    def forward(self, x):
        device = self.diag_mu.device

        diag_eta = torch.randn(self.size, device=device)
        tril_eta = torch.randn(self.tril_size, device=device)

        zeta_diag = self.diag_mu + diag_eta * self.diag_omega.exp()
        zeta_tril = self.tril_mu + tril_eta * self.tril_omega.exp()

        chol = torch.zeros(self.size, self.size, device=device)
        chol[self.diag_idx, self.diag_idx] = zeta_diag
        chol[self.tril_idx[0], self.tril_idx[1]] = zeta_tril
        theta = torch.matmul(chol, chol.t())

        constraint_contrib = self.diag_omega.sum() + self.tril_omega.sum() + self.diag_scale * zeta_diag
        constraint_contrib = Constraint.new(self.name, constraint_contrib)

        return theta, constraint_contrib

    def sample(self, x, size) -> torch.Tensor:
        device = self.diag_mu.device

        diag_sample_size = [size, self.size]
        diag_eta = torch.randn(diag_sample_size, device=device)
        diag_mu = self.diag_mu.detach().unsqueeze(0)
        diag_omega = self.diag_omega.detach().unsqueeze(0)
        diag_zeta = diag_mu + diag_eta * diag_omega.exp()

        tril_sample_size = [size, self.tril_size]
        tril_eta = torch.randn(tril_sample_size, device=device)
        tril_mu = self.tril_mu.detach().unsqueeze(0)
        tril_omega = self.tril_omega.detach().unsqueeze(0)
        tril_zeta = tril_mu + tril_eta * tril_omega.exp()

        chol = torch.zeros(size, self.size, self.size, device=device, dtype=torch.float64)
        chol[:, self.diag_idx, self.diag_idx] = diag_zeta
        chol[:, self.tril_idx[0], self.tril_idx[1]] = tril_zeta

        theta = torch.bmm(chol, chol.permute(0, 2, 1))

        return theta

    def extra_repr(self) -> str:
        return f'name={self.name}, size={self.size}'
