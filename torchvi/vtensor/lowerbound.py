import torch

from torchvi.core.vmodule import VModule
from torchvi.vtensor import utils
from torchvi.vtensor.backing import Backing
from torchvi.vtensor.constraint import Constraint


class LowerBound(VModule):
    def __init__(self, size, lower_bound, name: str = None):
        super().__init__(name)
        self.backing = Backing(size=size, name=f'{self.name}_backing')
        utils.check_numeric(lower_bound, 'lower_bound')
        self.lower_bound = lower_bound

    def forward(self, x):
        zeta, constraint_contrib = self.backing.forward()
        constraint_contrib += Constraint.new(f'{self.backing.name}_lb', zeta.sum())

        return self.lower_bound + zeta.exp(), constraint_contrib

    def sample(self, x, size) -> torch.Tensor:
        zeta = self.backing.sample(size)
        return self.lower_bound + torch.exp(zeta)

    def extra_repr(self) -> str:
        return f'name={self.name}, lower_bound={self.lower_bound}'
