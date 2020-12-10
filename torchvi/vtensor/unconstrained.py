from torchvi.vmodule import VModule
from torchvi.vtensor.backing import Backing


class Unconstrained(VModule):
    def __init__(self, size, name=None):
        super().__init__()
        self.backing = Backing(size, name)

    def forward(self, x):
        return self.backing.forward()

    def sample(self, x, size):
        return self.backing.sample(size)

    def extra_repr(self):
        return f'name={self.backing.name}, size={self.backing.size}'
