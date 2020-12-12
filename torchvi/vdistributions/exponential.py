from functools import cached_property
from typing import Set
from torch import distributions

from torchvi.core.ast import ASTNode, ArgsDict, SamplesDict
from torchvi.core.constant import wrap_if_constant
from torchvi.core.constraint import Constraint
from torchvi.core.vmodule import VModule
from torchvi.vtensor.lowerbound import LowerBoundImpl


class ExponentialNode(ASTNode):
    def __init__(self, node_name: str, rate_name: str, name: str):
        super().__init__(name)
        self.__node_name = node_name
        self.__rate_name = rate_name

    @property
    def node_name(self) -> str:
        return self.__node_name

    @property
    def rate_name(self) -> str:
        return self.__rate_name

    @cached_property
    def dependencies(self) -> Set[str]:
        return set([self.node_name, self.rate_name])

    def __call__(self, xs, args: ArgsDict):
        zeta, constraint_contrib = args[self.node_name]
        rate, rate_constraint = args[self.rate_name]

        prior = distributions.Exponential(rate)
        prior_constraint = Constraint.new(f'{self.name}_prior', prior.log_prob(zeta).sum())

        constraint_contrib += rate_constraint + prior_constraint

        args[self.name] = (zeta, constraint_contrib)

    def sample(self, xs, samples: SamplesDict):
        samples[self.name] = samples[self.node_name]

    def __repr__(self) -> str:
        return f'Exponential(name: {self.name}, node_name: {self.node_name}, rate_name: {self.rate_name})'


class Exponential(VModule):
    def __init__(self, size, rate, name: str):
        super().__init__(name=name)
        node_name = f'{self.name}_node'
        rate_name = f'{self.name}_rate'

        self._module_dict[node_name] = LowerBoundImpl(size=size, lower_bound=0.0, name=node_name)
        rate = wrap_if_constant(rate, name=rate_name)

        self._module_dict.update(rate._module_dict)
        self._graph_dict.update(rate._graph_dict)

        terminal_node = ExponentialNode(node_name=node_name, rate_name=rate_name, name=name)
        self._terminal_node = terminal_node
        self._graph_dict[self.name] = terminal_node
