from torch import distributions

from torchvi.core.vmodule import VModule
from torchvi.vtensor.lowerupperbound import LowerUpperBound
from torchvi.vtensor.constraint import Constraint
from torchvi.vdistributions.constant import wrap_if_constant


class Beta(VModule):
    def __init__(self, size, alpha, beta, name=None):
        super().__init__()
        self.size = size
        self.node = LowerUpperBound(size=size, lower_bound=0.0, upper_bound=0.0, name=name)
        name = self.node.backing.name
        self.alpha = wrap_if_constant(alpha, name=f'{name}_alpha')
        self.beta = wrap_if_constant(beta, name=f'{name}_beta')

    def forward(self, x):
        zeta, constraint_contrib = self.node.forward(x)

        alpha, alpha_constraint = self.alpha.forward(x)
        beta, beta_constraint = self.beta.forward(x)
        constraint_contrib += alpha_constraint + beta_constraint

        prior = distributions.Beta(alpha, beta)
        name = self.node.backing.name
        constraint_contrib += Constraint.new(f'{name}_prior', prior.log_prob(zeta).sum())

        return zeta, constraint_contrib

    def sample(self, x, size):
        return self.node.sample(x, size)

    def extra_repr(self):
        return f'size={self.size}, alpha={self.alpha}, beta={self.beta}'
