from collections.abc import Iterable
from functools import reduce
from torchvi.vtensor import utils
from torch import nn
import torch

from torchvi.core.ast import SingleNodeIdentity
from torchvi.core.vmodule import VModule
from torchvi.vtensor.lowerbound import LowerBoundImpl
from torchvi.vtensor.unconstrained import UnconstrainedImpl


class CholeskyImpl(nn.Module):
    def __init__(self, size, name: str):
        super().__init__()
        self.name = name

        if isinstance(size, int):
            self.batch_size = [1]
            self.mat_size = size
            self.squeeze_batch = True
        elif isinstance(size, Iterable):
            if len(size) < 2:
                raise ValueError('The length of the size iterable must be at least 2')
            self.batch_size = size[:-1]
            self.mat_size = size[-1]
            self.squeeze_batch = False
        else:
            raise TypeError(f'size must be either an int or an iterable of size 2 of ints. Got: {type(size)}')
        self.batch_size_prod = reduce(lambda x, y: x * y, self.batch_size)
        self.res_size = self.batch_size + [self.mat_size, self.mat_size]

        if self.mat_size <= 1:
            raise ValueError(f'The size of the cholesky matrix must be greater than 1. Got: {self.mat_size}')

        diag_idx = torch.arange(self.mat_size)
        tril_idx = torch.tril_indices(row=self.mat_size, col=self.mat_size, offset=-1)

        self.register_buffer('diag_idx', diag_idx)
        self.register_buffer('tril_idx', tril_idx)

        diag_size = [self.batch_size_prod, self.mat_size]
        tril_size = [self.batch_size_prod, (self.mat_size * (self.mat_size - 1)) // 2]

        self.diag = LowerBoundImpl(size=diag_size, lower_bound=0.0, name=f'{self.name}_diag')
        self.tril = UnconstrainedImpl(size=tril_size, name=f'{self.name}_tril')

    def forward(self):
        diag_zeta, diag_contrib = self.diag()
        tril_zeta, tril_contrib = self.tril()

        theta = torch.zeros(self.batch_size_prod, self.mat_size, self.mat_size, device=diag_zeta.device)
        theta[:, self.diag_idx, self.diag_idx] = diag_zeta
        theta[:, self.tril_idx[0], self.tril_idx[1]] = tril_zeta

        if self.squeeze_batch:
            theta = theta.squeeze(0)
        else:
            theta = theta.view(self.res_size)

        constraint_contrib = diag_contrib + tril_contrib

        return theta, constraint_contrib

    def sample(self, size) -> torch.Tensor:
        size = list(utils.to_size(size))
        size_prod = reduce(lambda x, y: x * y, size)
        diag_samples = self.diag.sample(size_prod)
        tril_samples = self.tril.sample(size_prod)

        calc_size = [size_prod, self.batch_size_prod, self.mat_size, self.mat_size]
        view_size = size + self.batch_size + [self.mat_size, self.mat_size]
        squeeze_idx = len(size)

        chol = torch.zeros(calc_size, device=diag_samples.device)
        chol[:, :, self.diag_idx, self.diag_idx] = diag_samples
        chol[:, :, self.tril_idx[0], self.tril_idx[1]] = tril_samples
        chol = chol.view(view_size)

        if self.squeeze_batch:
            chol = chol.squeeze(squeeze_idx)

        return chol

    def extra_repr(self) -> str:
        return f'name={self.name}'


class Cholesky(VModule):
    def __init__(self, size: int, name: str):
        super().__init__(name)
        impl_name = f'{self.name}_impl'
        self._module_dict[impl_name] = CholeskyImpl(size=size, name=impl_name)
        terminal_node = SingleNodeIdentity(name=self.name, arg=impl_name)
        self._terminal_node = terminal_node
        self._graph_dict[self.name] = terminal_node
