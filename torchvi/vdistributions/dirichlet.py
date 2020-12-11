from collections.abc import Iterable
from torchvi.vdistributions.constant import wrap_if_constant
from torch import distributions

from torchvi.core.vmodule import VModule
from torchvi.vtensor.constraint import Constraint
from torchvi.vtensor.simplex import Simplex
from torchvi.vdistributions.constant import wrap_if_constant


class Dirichlet(VModule):
    def __init__(self, alpha, name=None):
        super().__init__()
        if isinstance(alpha, Iterable):
            self.size = len(alpha)
        else:
            raise TypeError(f'alpha must be an iterable. Got: {type(alpha)}')
        self.size = len(alpha)
        self.node = Simplex(size=self.size, name=name)
        self.alpha = wrap_if_constant(alpha, name=f'{self.node.backing.name}_alpha')

    def forward(self, x):
        theta, constraint_contrib = self.node.forward(x)
        alpha, alpha_constraint = self.alpha.forward(x)

        prior = distributions.Dirichlet(alpha)
        name = self.node.backing.name
        constraint_contrib += alpha_constraint + Constraint.new(f'{name}_prior', prior.log_prob(theta).sum())

        return theta, constraint_contrib

    def sample(self, x, size):
        return self.node.sample(x, size)

    def extra_repr(self):
        return f'size={self.size}'
