from functools import cached_property
from typing import Set
from torch import distributions

from torchvi.core.ast import ASTNode, ArgsDict, SamplesDict
from torchvi.core.constant import wrap_if_constant
from torchvi.core.constraint import Constraint
from torchvi.core.vmodule import VModule
from torchvi.vtensor.lowerbound import LowerBoundImpl


class HalfNormalNode(ASTNode):
    def __init__(self, node_name: str, scale_name: str, name: str):
        super().__init__(name)
        self.__node_name = node_name
        self.__scale_name = scale_name

    @property
    def node_name(self):
        return self.__node_name

    @property
    def scale_name(self):
        return self.__scale_name

    @cached_property
    def dependencies(self) -> Set[str]:
        return set([self.node_name, self.scale_name])

    def __call__(self, xs, args: ArgsDict):
        zeta, constraint_contrib = args[self.node_name]
        scale, scale_constraint = args[self.scale_name]
        constraint_contrib += scale_constraint

        prior = distributions.HalfNormal(scale)
        constraint_contrib += Constraint.new(f'{self.name}_prior', prior.log_prob(zeta).sum())

        args[self.name] = (zeta, constraint_contrib)

    def sample(self, xs, samples: SamplesDict):
        samples[self.name] = samples[self.node_name]

    def __repr__(self) -> str:
        return f'HalfNormal(name: {self.name}, node: {self.node_name}, scale: {self.scale_name})'


class HalfNormal(VModule):
    def __init__(self, size, scale, name: str):
        super().__init__(name=name)
        node_name = f'{self.name}_node'
        scale_name = f'{self.name}_scale'

        self._module_dict[node_name] = LowerBoundImpl(size=size, lower_bound=0.0, name=node_name)
        scale = wrap_if_constant(scale, name=scale_name)

        self._module_dict.update(scale._module_dict)
        self._graph_dict.update(scale._graph_dict)

        terminal_node = HalfNormalNode(node_name=node_name, scale_name=scale_name, name=self.name)
        self._terminal_node = terminal_node
        self._graph_dict[self.name] = terminal_node
