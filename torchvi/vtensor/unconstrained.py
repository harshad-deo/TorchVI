import torch

from torchvi.core.vmodule import VModule
from torchvi.vtensor.backing import Backing


class Unconstrained(VModule):
    def __init__(self, size, name=None):
        super().__init__(name=name)
        self.backing = Backing(size, f'{self.name}_backing')

    def forward(self, x):
        return self.backing.forward()

    def sample(self, x, size) -> torch.Tensor:
        return self.backing.sample(size)

    def extra_repr(self) -> str:
        return f'name={self.name}'
