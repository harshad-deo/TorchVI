import torch
import uuid

from torchvi.vmodule import VModule
from torchvi.vtensor.unconstrained import Unconstrained
from torchvi.vtensor.lowerbound import LowerBound


class Cholesky(VModule):
    def __init__(self, size: int, name=None):
        super().__init__()

        if not isinstance(size, int):
            raise TypeError(f'size must be an integer. Got: {type(size)}')
        if size <= 1:
            raise ValueError(f'size must be greater than 1. Got: {size}')

        if name is None:
            name = uuid.uuid4().hex
        self.__name = name

        self.size = size
        self.tril_size = (size * (size - 1)) // 2

        diag_idx = torch.arange(size)
        tril_idx = torch.tril_indices(row=size, col=size, offset=-1)
        self.register_buffer('diag_idx', diag_idx)
        self.register_buffer('tril_idx', tril_idx)

        self.diag = LowerBound(size=self.size, lower_bound=0.0, name=f'{name}_diag')
        self.tril = Unconstrained(size=self.tril_size, name=f'{name}_tril')

    def forward(self, x):
        diag_zeta, diag_contrib = self.diag.forward(x)
        tril_zeta, tril_contrib = self.tril.forward(x)

        theta = torch.zeros(self.size, self.size, device=diag_zeta.device)
        theta[self.diag_idx, self.diag_idx] = diag_zeta
        theta[self.tril_idx[0], self.tril_idx[1]] = tril_zeta

        constraint_contrib = diag_contrib + tril_contrib

        return theta, constraint_contrib

    def sample(self, x, size):
        diag_samples = self.diag.sample(x, size)
        tril_samples = self.tril.sample(x, size)

        chol = torch.zeros([size, self.size, self.size], device=diag_samples.device)
        chol[:, self.diag_idx, self.diag_idx] = diag_samples
        chol[:, self.tril_idx[0], self.tril_idx[1]] = tril_samples

        return chol

    @property
    def name(self) -> str:
        return self.__name

    def extra_repr(self):
        return f'name={self.__name} size={self.size}'
