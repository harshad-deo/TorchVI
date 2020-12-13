from collections.abc import Iterable
from functools import cached_property
from torch import distributions
from typing import Set

from torchvi.core.ast import ASTNode, ArgsDict, SamplesDict
from torchvi.core.constraint import Constraint
from torchvi.core.vmodule import VModule, wrap_if_constant
from torchvi.vtensor.simplex import SimplexImpl


class DirichletNode(ASTNode):
    def __init__(self, node_name: str, alpha_name: str, name: str):
        super().__init__(name)
        self.__node_name = node_name
        self.__alpha_name = alpha_name

    @property
    def node_name(self) -> str:
        return self.__node_name

    @property
    def alpha_name(self) -> str:
        return self.__alpha_name

    @cached_property
    def dependencies(self) -> Set[str]:
        return set([self.node_name, self.alpha_name])

    def __call__(self, xs, args: ArgsDict):
        theta, constraint_contrib = args[self.node_name]
        alpha, alpha_constraint = args[self.alpha_name]

        prior = distributions.Dirichlet(alpha)
        constraint_contrib += alpha_constraint + Constraint.new(f'{self.name}_prior', prior.log_prob(theta).sum())

        args[self.name] = (theta, constraint_contrib)

    def sample(self, xs, samples: SamplesDict):
        samples[self.name] = samples[self.node_name]

    def __repr__(self) -> str:
        return f'Dirichlet(name: {self.name}, node: {self.node_name}, alpha: {self.alpha_name})'


class Dirichlet(VModule):
    def __init__(self, alpha, name: str):
        super().__init__(name)
        if isinstance(alpha, Iterable):
            self.size = len(alpha)
        else:
            raise TypeError(f'alpha must be an iterable. Got: {type(alpha)}')

        node_name = f'{self.name}_node'
        alpha_name = f'{self.name}_alpha'

        self.size = len(alpha)
        self._module_dict[node_name] = SimplexImpl(size=self.size, name=f'{self.name}_node')
        alpha = wrap_if_constant(alpha, name=alpha_name)

        self._module_dict.update(alpha._module_dict)
        self._graph_dict.update(alpha._graph_dict)

        terminal_node = DirichletNode(node_name=node_name, alpha_name=alpha.name, name=self.name)
        self._terminal_node = terminal_node
        self._graph_dict[self.name] = terminal_node
