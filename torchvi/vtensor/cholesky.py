from torch import nn
import torch

from torchvi.core.ast import SingleNodeIdentity
from torchvi.core.vmodule import VModule
from torchvi.vtensor.lowerbound import LowerBoundImpl
from torchvi.vtensor.unconstrained import UnconstrainedImpl


class CholeskyImpl(nn.Module):
    def __init__(self, size: int, name: str):
        super().__init__()

        if not isinstance(size, int):
            raise TypeError(f'size must be an integer. Got: {type(size)}')
        if size <= 1:
            raise ValueError(f'size must be greater than 1. Got: {size}')

        self.name = name
        self.size = size
        self.tril_size = (size * (size - 1)) // 2

        diag_idx = torch.arange(size)
        tril_idx = torch.tril_indices(row=size, col=size, offset=-1)
        self.register_buffer('diag_idx', diag_idx)
        self.register_buffer('tril_idx', tril_idx)

        self.diag = LowerBoundImpl(size=size, lower_bound=0.0, name=f'{self.name}_diag')
        self.tril = UnconstrainedImpl(size=self.tril_size, name=f'{self.name}_tril')

    def forward(self):
        diag_zeta, diag_contrib = self.diag()
        tril_zeta, tril_contrib = self.tril()

        theta = torch.zeros(self.size, self.size, device=diag_zeta.device)
        theta[self.diag_idx, self.diag_idx] = diag_zeta
        theta[self.tril_idx[0], self.tril_idx[1]] = tril_zeta

        constraint_contrib = diag_contrib + tril_contrib

        return theta, constraint_contrib

    def sample(self, size) -> torch.Tensor:
        diag_samples = self.diag.sample(size)
        tril_samples = self.tril.sample(size)

        chol = torch.zeros([size, self.size, self.size], device=diag_samples.device)
        chol[:, self.diag_idx, self.diag_idx] = diag_samples
        chol[:, self.tril_idx[0], self.tril_idx[1]] = tril_samples

        return chol

    def extra_repr(self) -> str:
        return f'name={self.name}, size={self.size}'


class Cholesky(VModule):
    def __init__(self, size: int, name: str):
        super().__init__(name)
        impl_name = f'{self.name}_impl'
        self._module_dict[impl_name] = CholeskyImpl(size=size, name=impl_name)
        terminal_node = SingleNodeIdentity(name=self.name, arg=impl_name)
        self._terminal_node = terminal_node
        self._graph_dict[self.name] = terminal_node
