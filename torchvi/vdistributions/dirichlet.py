from collections.abc import Iterable
from torchvi.vdistributions.constant import wrap_if_constant
from torch import distributions

from torchvi.vmodule import VModule
from torchvi.vtensor.simplex import Simplex
from torchvi.vdistributions.constant import wrap_if_constant




class Dirichlet(VModule):
    def __init__(self, alpha):
        super().__init__()
        if isinstance(alpha, Iterable):
            self.size = len(alpha)
        else:
            raise TypeError(f'alpha must be an iterable. Got: {type(alpha)}')
        self.size = len(alpha)
        self.alpha = wrap_if_constant(alpha)
        self.node = Simplex(size=self.size)

    def forward(self, x):
        theta, constraint_contrib = self.node.forward(x)
        alpha, alpha_constraint = self.alpha.forward(x)

        prior = distributions.Dirichlet(alpha)
        constraint_contrib += alpha_constraint + prior.log_prob(theta).sum()

        return theta, constraint_contrib

    def sample(self, x, size):
        return self.node.sample(x, size)

    def extra_repr(self):
        return f'size={self.size}'