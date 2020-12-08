import torch
import torch.nn.functional as F

from torchvi.vtensor.unconstrained import Unconstrained


class Simplex(Unconstrained):
    def __init__(self, size: int):
        super().__init__(size - 1)
        if not isinstance(size, int):
            raise TypeError(f'Simplex size must be an integer. Got: {type(size)}')
        if size <= 1:
            raise ValueError(f'Simplex size must be greater than 1. Got: {size}')
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
        zeta, constraint_contrib = super().forward(x)
        z, theta = self.__simplex_transform(zeta)

        simplex_constraint_contrib = z.log() + self.grad_scale * (1 - z).log()

        constraint_contrib += simplex_constraint_contrib.sum()
        return theta, constraint_contrib

    def sample(self, x, size):
        zeta = super().sample(x, size)
        _, theta = self.__simplex_transform(zeta)
        return theta
