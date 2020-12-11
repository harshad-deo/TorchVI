from collections.abc import Iterable
import torch
from torch import distributions

from torchvi.core.constant import wrap_if_constant
from torchvi.core.constraint import Constraint
from torchvi.core.vmodule import VModule
from torchvi.vtensor.simplex import Simplex


class Dirichlet(VModule):
    def __init__(self, alpha, name: str):
        super().__init__(name)
        if isinstance(alpha, Iterable):
            self.size = len(alpha)
        else:
            raise TypeError(f'alpha must be an iterable. Got: {type(alpha)}')
        self.size = len(alpha)
        self.node = Simplex(size=self.size, name=f'{self.name}_node')
        self.alpha = wrap_if_constant(alpha, name=f'{self.name}_alpha')

    def forward(self, x):
        theta, constraint_contrib = self.node.forward(x)
        alpha, alpha_constraint = self.alpha.forward(x)

        prior = distributions.Dirichlet(alpha)
        name = self.node.backing.name
        constraint_contrib += alpha_constraint + Constraint.new(f'{name}_prior', prior.log_prob(theta).sum())

        return theta, constraint_contrib

    def sample(self, x, size) -> torch.Tensor:
        return self.node.sample(x, size)

    def extra_repr(self) -> str:
        return f'name={self.name} size={self.size}'
