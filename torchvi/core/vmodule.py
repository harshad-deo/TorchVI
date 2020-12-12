from abc import ABC
from collections import deque
from functools import cached_property
import logging
from torchvi.core.constraint import Constraint
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from torchvi.core.ast import ASTNode, ArgsDict

log = logging.getLogger(__name__)


class VModule(nn.Module, ABC):
    def __init__(self, name: str):
        super().__init__()
        self.__name = name
        self._module_dict = nn.ModuleDict()
        self._graph_dict: Dict[str, ASTNode] = {}
        self._terminal_node: Optional[ASTNode] = None

    @cached_property
    def __node_seq(self) -> List[str]:
        if self._terminal_node is None:
            raise ValueError('Invariant failure: terminal node not set')
        ctr = 0
        resolved: Dict[str, int] = {}
        deps_pending = deque([self._terminal_node.name])
        while len(deps_pending) > 0:
            elem_name = deps_pending.popleft()
            if elem_name in resolved:
                continue
            if elem_name in self._module_dict:
                resolved[elem_name] = ctr
                ctr += 1
                continue
            elem = self._graph_dict[elem_name]
            unresolved = [y for y in elem.dependencies if y not in resolved]
            if len(unresolved) > 0:
                deps_pending.appendleft(elem_name)
                deps_pending.extendleft(unresolved)
            else:
                resolved[elem.name] = ctr
                ctr += 1
        resolved_inv = {v: k for k, v in resolved.items()}
        node_seq = [resolved_inv[x] for x in range(ctr)]
        # Invariants:
        #   1. Each element is evaluated only once
        #   2. Each element is evaluated only after all its dependencies are evaluated
        log.info(f'Calculated node_seq: {node_seq}')
        return node_seq

    def forward(self, xs) -> Tuple[torch.Tensor, Constraint]:
        node_seq = self.__node_seq
        args_dict: ArgsDict = {}
        for node_name in node_seq:
            if node_name in self._module_dict:
                args_dict[node_name] = self._module_dict[node_name]()
                continue
            node = self._graph_dict[node_name]
            node(xs, args_dict)
        return args_dict[self._terminal_node.name]

    def sample(self, xs, size) -> torch.Tensor:
        node_seq = self.__node_seq
        samples_dict = {k: v.sample(size) for k, v in self._module_dict.items()}
        for node_name in node_seq:
            if node_name in self._module_dict:
                samples_dict[node_name] = self._module_dict[node_name].sample(size)
                continue
            node = self._graph_dict[node_name]
            node.sample(xs, samples_dict)
        return samples_dict[self._terminal_node.name]

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def name(self) -> str:
        return self.__name

    def extra_repr(self) -> str:
        return f'name={self.name}'
