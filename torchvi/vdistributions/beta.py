from functools import cached_property
import torch
from torch import distributions
from typing import Set

from torchvi.core.ast import ASTNode, ArgsDict, SamplesDict
from torchvi.core.constraint import Constraint
from torchvi.core.vmodule import VModule, wrap_if_constant
from torchvi.vtensor.lowerupperbound import LowerUpperBoundImpl


class BetaNode(ASTNode):
    def __init__(self, node_name: str, alpha_name: str, beta_name: str, name: str):
        super().__init__(name)
        self.__node_name = node_name
        self.__alpha_name = alpha_name
        self.__beta_name = beta_name

    def __call__(self, xs, args: ArgsDict):
        zeta, constraint_contrib = args[self.node_name]

        alpha, alpha_constraint = args[self.alpha_name]
        beta, beta_constraint = args[self.beta_name]
        constraint_contrib += alpha_constraint + beta_constraint

        prior = distributions.Beta(alpha, beta)
        constraint_contrib += Constraint.new(f'{self.name}_prior', prior.log_prob(zeta).sum())

        args[self.name] = (zeta, constraint_contrib)

    def sample(self, xs, samples: SamplesDict):
        samples[self.name] = samples[self.node_name]

    @property
    def node_name(self):
        return self.__node_name

    @property
    def alpha_name(self):
        return self.__alpha_name

    @property
    def beta_name(self):
        return self.__beta_name

    @cached_property
    def dependencies(self) -> Set[str]:
        return set([self.node_name, self.alpha_name, self.beta_name])

    def __repr__(self) -> str:
        return f'Beta(name: {self.name}, node: {self.node_name}, alpha: {self.alpha_name}, beta: {self.beta_name})'


class Beta(VModule):
    def __init__(self, size, alpha, beta, name: str):
        super().__init__(name=name)
        node_name = f'{self.name}_node'
        alpha_name = f'{self.name}_alpha'
        beta_name = f'{self.name}_beta'

        self._module_dict[node_name] = LowerUpperBoundImpl(size=size, lower_bound=0.0, upper_bound=1.0, name=node_name)

        alpha = wrap_if_constant(alpha, name=alpha_name)
        beta = wrap_if_constant(beta, name=beta_name)

        self._module_dict.update(alpha._module_dict)
        self._module_dict.update(beta._module_dict)

        self._graph_dict.update(alpha._graph_dict)
        self._graph_dict.update(beta._graph_dict)

        terminal_node = BetaNode(node_name=node_name, alpha_name=alpha.name, beta_name=beta.name, name=name)
        self._terminal_node = terminal_node
        self._graph_dict[self.name] = terminal_node
