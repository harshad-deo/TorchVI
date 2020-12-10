import torch
import torch.nn.functional as F

from torchvi.vtensor.backing import Backing
from torchvi.vmodule import VModule


class Simplex(VModule):
    def __init__(self, size: int, name=None):
        super().__init__()

        if not isinstance(size, int):
            raise TypeError(f'Simplex size must be an integer. Got: {type(size)}')
        if size <= 1:
            raise ValueError(f'Simplex size must be greater than 1. Got: {size}')

        self.backing = Backing(size - 1, name)
        ar = torch.arange(1, size, dtype=torch.float64, requires_grad=False)
        grad_scale = size - ar
        self.register_buffer('grad_scale', grad_scale)
        log_add = -grad_scale.log()
        self.register_buffer('log_add', log_add)

    def __simplex_transform(self, zeta):
        z = torch.sigmoid(zeta + self.log_add)
        zl = F.pad(input=z, pad=[1, 0], value=0)  # left pad
        zr = F.pad(input=z, pad=[0, 1], value=1)  # right pad
        theta = (1 - zl).cumprod(-1) * zr
        return z, theta

    def forward(self, x):
        zeta, constraint_contrib = self.backing.forward()
        z, theta = self.__simplex_transform(zeta)

        simplex_constraint_contrib = z.log() + self.grad_scale * (1 - z).log()

        constraint_contrib += simplex_constraint_contrib.sum()
        return theta, constraint_contrib

    def sample(self, x, size):
        zeta = self.backing.sample(size)
        _, theta = self.__simplex_transform(zeta)
        return theta

    def extra_repr(self):
        return f'name={self.backing.name} size={self.backing.size + 1}'
