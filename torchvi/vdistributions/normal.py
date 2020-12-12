from functools import cached_property
from typing import Set
from torch import distributions

from torchvi.core.ast import ASTNode, ArgsDict, SamplesDict
from torchvi.core.constant import wrap_if_constant
from torchvi.core.vmodule import VModule
from torchvi.core.constraint import Constraint
from torchvi.vtensor.unconstrained import UnconstrainedImpl


class NormalNode(ASTNode):
    def __init__(self, node_name: str, loc_name: str, scale_name: str, name: str):
        super().__init__(name)
        self.__node_name = node_name
        self.__loc_name = loc_name
        self.__scale_name = scale_name

    @property
    def node_name(self) -> str:
        return self.__node_name

    @property
    def loc_name(self) -> str:
        return self.__loc_name

    @property
    def scale_name(self) -> str:
        return self.__scale_name

    @cached_property
    def dependencies(self) -> Set[str]:
        return set([self.node_name, self.loc_name, self.scale_name])

    def __call__(self, xs, args: ArgsDict):
        zeta, constraint_contrib = args[self.node_name]
        loc, loc_constraint = args[self.loc_name]
        scale, scale_constraint = args[self.scale_name]

        prior = distributions.Normal(loc=loc, scale=scale)
        prior_constraint = Constraint.new(f'{self.name}_prior', prior.log_prob(zeta).sum())
        constraint_contrib += prior_constraint + loc_constraint + scale_constraint

        args[self.name] = (zeta, constraint_contrib)

    def sample(self, xs, samples: SamplesDict):
        samples[self.name] = samples[self.node_name]

    def __repr__(self) -> str:
        return f'Normal(name: {self.name}, node: {self.node_name}, loc: {self.loc_name}, scale: {self.scale_name})'


class Normal(VModule):
    def __init__(self, size, loc, scale, name: str):
        super().__init__(name)

        node_name = f'{self.name}_node'
        loc_name = f'{self.name}_loc'
        scale_name = f'{self.name}_scale'

        self._module_dict[node_name] = UnconstrainedImpl(size=size, name=node_name)
        loc = wrap_if_constant(loc, name=loc_name)
        scale = wrap_if_constant(scale, name=scale_name)

        self._module_dict.update(loc._module_dict)
        self._module_dict.update(scale._module_dict)

        self._graph_dict.update(loc._graph_dict)
        self._graph_dict.update(scale._graph_dict)

        terminal_node = NormalNode(node_name=node_name, loc_name=loc_name, scale_name=scale_name, name=self.name)
        self._terminal_node = terminal_node
        self._graph_dict[self.name] = terminal_node
