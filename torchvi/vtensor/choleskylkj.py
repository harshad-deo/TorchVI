from collections.abc import Iterable
from functools import reduce
import torch
from torch import nn

from torchvi.core.constraint import Constraint
from torchvi.core.vmodule import VModule
from torchvi.core.ast import SingleNodeIdentity
from torchvi.vtensor.unconstrained import UnconstrainedImpl
from torchvi.vtensor import utils


class CholeskyLKJImpl(nn.Module):
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

        if self.mat_size <= 1:
            raise ValueError(f'The size of the cholesky matrix must be greater than 1. Got: {self.mat_size}')

        diag_idx = torch.arange(self.mat_size)
        tril_idx = torch.tril_indices(row=self.mat_size, col=self.mat_size, offset=-1)

        self.register_buffer('diag_idx', diag_idx)
        self.register_buffer('tril_idx', tril_idx)

        tril_size = [self.batch_size_prod, (self.mat_size * (self.mat_size - 1)) // 2]
        self.tril = UnconstrainedImpl(size=tril_size, name=f'{self.name}_tril')

    def forward(self):
        tril, tril_constraint = self.tril()
        device = tril.device

        tril_tanh = tril.tanh()
        z = torch.zeros(self.batch_size_prod, self.mat_size, self.mat_size, device=device)
        z[:, self.tril_idx[0], self.tril_idx[1]] = tril_tanh

        lkj_contraint = tril.cosh().log().sum() * -2

        for i in range(self.mat_size):
            cumsum = torch.zeros(self.batch_size_prod, device=device)
            for j in range(i):
                x = z[:, i, j]
                x = x * torch.sqrt(1 - cumsum)
                z[:, i, j] = x
                lkj_contraint += 0.5 * torch.log(1 - cumsum).sum()
                cumsum += x * x
            z[:, i, i] = torch.sqrt(1 - cumsum)

        if self.squeeze_batch:
            z = torch.squeeze(z, 0)

        lkj_contraint = Constraint.new(f'{self.name}_lkj', lkj_contraint)
        constraint = lkj_contraint + tril_constraint
        return z, constraint

    def sample(self, size):
        size = list(utils.to_size(size))
        size_prod = reduce(lambda x, y: x * y, size)
        tril_samples = self.tril.sample(size_prod)

        tril_samples = self.tril.sample(size_prod)
        device = tril_samples.device

        calc_size = [size_prod, self.batch_size_prod, self.mat_size, self.mat_size]
        view_size = size + self.batch_size + [self.mat_size, self.mat_size]
        squeeze_idx = len(size)

        z = torch.zeros(calc_size, device=device)
        z[:, :, self.tril_idx[0], self.tril_idx[1]] = tril_samples.tanh()

        for i in range(self.mat_size):
            cumsum = torch.zeros([size_prod, self.batch_size_prod], device=device)
            for j in range(i):
                x = z[:, :, i, j]
                x = x * torch.sqrt(1 - cumsum)
                z[:, :, i, j] = x
                cumsum += x * x
            z[:, :, i, i] = torch.sqrt(1 - cumsum)

        z = z.view(view_size)

        if self.squeeze_batch:
            z = torch.squeeze(z, squeeze_idx)

        return z


class CholeskyLKJ(VModule):
    def __init__(self, size, name: str):
        super().__init__(name)
        impl_name = f'{self.name}_impl'
        self._module_dict[impl_name] = CholeskyLKJImpl(size=size, name=impl_name)
        terminal_node = SingleNodeIdentity(name=self.name, arg=impl_name)
        self._terminal_node = terminal_node
        self._graph_dict[self.name] = terminal_node
