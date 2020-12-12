from torchvi.core.ast import SingleNodeIdentity
from torchvi.core.vmodule import VModule
from torchvi.vtensor.backing import Backing


class Unconstrained(VModule):
    def __init__(self, size, name: str):
        super().__init__(name=name)
        backing_name = f'{self.name}_backing'
        self._module_dict[backing_name] = Backing(size=size, name=backing_name)
        terminal_node = SingleNodeIdentity(name=self.name, arg=backing_name)
        self._terminal_node = terminal_node
        self._graph_dict[self.name] = terminal_node
