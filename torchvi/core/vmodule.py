from abc import ABC
from collections import deque
from functools import cached_property
import hashlib
import logging
from torchvi.core.constraint import Constraint
from typing import Dict, List, Optional, Tuple, Iterable, Mapping
import torch
from torch import nn

from torchvi.core.ast import *

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

    def __neg__(self):
        add_inv_name = f'additive_inv_{self.name}'
        terminal_node = AdditiveInverseNode(name=add_inv_name, arg=self.name)
        res = Vop(modules=self._module_dict, graph=self._graph_dict, terminal_node=terminal_node, name=add_inv_name)
        return res

    def __add__(self, that):
        rhs_name = f'{self.name}_add'
        that = wrap_if_constant(that, rhs_name)
        add_name = f'add_{self.name}_{that.name}'
        terminal_node = AdditionNode(name=add_name, lhs=self.name, rhs=that.rhs)
        res_modules = {**self._module_dict, **that._module_dict}
        res_graph = {**self._graph_dict, **that._graph_dict}
        res = Vop(modules=res_modules, graph=res_graph, terminal_node=terminal_node, name=add_name)
        return res

    def __radd__(self, that):
        return self.__add__(that)

    def __sub__(self, that):
        rhs_name = f'{self.name}_sub_rhs'
        that = wrap_if_constant(that, rhs_name)
        sub_name = f'sub_{self.name}_{that.name}'
        terminal_node = SubtractionNode(name=sub_name, minuend=self.name, subtrahend=that.name)
        res_modules = {**self._module_dict, **that._module_dict}
        res_graph = {**self._graph_dict, **that._graph_dict}
        res = Vop(modules=res_modules, graph=res_graph, terminal_node=terminal_node, name=sub_name)
        return res

    def __rsub__(self, that):
        lhs_name = f'{self.name}_sub_lhs'
        that = wrap_if_constant(that, lhs_name)
        sub_name = f'sub_{that.name}_{self.name}'
        terminal_node = SubtractionNode(name=sub_name, minuend=that.name, subtrahend=self.name)
        res_modules = {**self._module_dict, **that._module_dict}
        res_graph = {**self._graph_dict, **that._graph_dict}
        res = Vop(modules=res_modules, graph=res_graph, terminal_node=terminal_node, name=sub_name)
        return res

    def __mul__(self, that):
        rhs_name = f'{self.name}_mul'
        that = wrap_if_constant(that, rhs_name)
        mul_name = f'mul_{self.name}_{that.name}'
        terminal_node = MultiplicationNode(name=mul_name, lhs=self.name, rhs=that.name)
        res_modules = {**self._module_dict, **that._module_dict}
        res_graph = {**self._graph_dict, **that._graph_dict}
        res = Vop(modules=res_modules, graph=res_graph, terminal_node=terminal_node, name=mul_name)
        return res

    def __rmul__(self, that):
        return self.__mul__(that)

    def __truediv__(self, that):
        rhs_name = f'{self.name}_div_rhs'
        that = wrap_if_constant(that, rhs_name)
        div_name = f'div_{self.name}_{that.name}'
        terminal_node = DivisionNode(name=div_name, dividend=self.name, divisor=that.name)
        res_modules = {**self._module_dict, **that._module_dict}
        res_graph = {**self._graph_dict, **that._graph_dict}
        res = Vop(modules=res_modules, graph=res_graph, terminal_node=terminal_node, name=div_name)
        return res

    def __rtruediv__(self, that):
        lhs_name = f'{self.name}_div_lhs'
        that = wrap_if_constant(that, lhs_name)
        div_name = f'div_{that.name}_{self.name}'
        terminal_node = DivisionNode(name=div_name, dividend=that.name, divisor=self.name)
        res_modules = {**self._module_dict, **that._module_dict}
        res_graph = {**self._graph_dict, **that._graph_dict}
        res = Vop(modules=res_modules, graph=res_graph, terminal_node=terminal_node, name=div_name)
        return res

    def exp(self):
        exp_name = f'{self.name}_exp'
        terminal_node = ExponentNode(name=exp_name, arg=self.name)
        res = Vop(modules=self._module_dict, graph=self._graph_dict, terminal_node=terminal_node, name=exp_name)
        return res

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def name(self) -> str:
        return self.__name

    def extra_repr(self) -> str:
        return f'name={self.name}'


class ConstantImpl(nn.Module):
    def __init__(self, value: torch.Tensor, name: str):
        super().__init__()
        self.name = name
        self.register_buffer('value', value)
        self.register_buffer('constraint_contrib', torch.squeeze(torch.zeros(1)))

    def forward(self):
        return self.value, Constraint.new(self.name, self.constraint_contrib)

    def sample(self, size) -> torch.Tensor:
        return self.value.repeat(size)

    def extra_repr(self) -> str:
        return f'name={self.name}, value={self.value}'


class Constant(VModule):
    def __init__(self, value: torch.Tensor, name: str):
        super().__init__(name=name)
        backing_name = f'{name}_backing'
        self._module_dict[backing_name] = ConstantImpl(value=value, name=backing_name)
        terminal_node = SingleNodeIdentity(name=self.name, arg=backing_name)
        self._terminal_node = terminal_node
        self._graph_dict[self.name] = terminal_node


# Variational Operation
class Vop(VModule):
    def __init__(self, modules: Mapping[str, nn.Module], graph: Mapping[str, ASTNode], terminal_node: ASTNode,
                 name: str):
        super().__init__(name)
        self._module_dict.update(modules)
        self._graph_dict.update(graph)
        self._terminal_node = terminal_node
        self._graph_dict[name] = terminal_node  # defensive check


def wrap_if_constant(x, name: str):
    if isinstance(x, VModule):
        return x
    if isinstance(x, int) or isinstance(x, float):
        tensor = torch.tensor(x, requires_grad=False)
    elif isinstance(x, Iterable):
        tensor = torch.tensor(x, requires_grad=False)
    elif isinstance(x, torch.Tensor):
        tensor = x
    else:
        raise TypeError(f'Unsupported type for wrapping. Expected int, float or tensor, got: {type(x)}')
    tensor_bytes = tensor.cpu().numpy().tobytes()
    m = hashlib.md5(name.encode('utf-8'))
    m.update(tensor_bytes)
    name = f'{name}_{m.hexdigest()}'
    return Constant(value=tensor, name=name)
