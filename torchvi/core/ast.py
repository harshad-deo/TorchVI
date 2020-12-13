from abc import ABC, abstractmethod
from functools import cached_property
from random import sample
from typing import Dict, Tuple, Set
import torch

from torchvi.core.constraint import Constraint

ArgsDict = Dict[str, Tuple[torch.Tensor, Constraint]]
SamplesDict = Dict[str, torch.Tensor]


class ASTNode(ABC):
    def __init__(self, name: str):
        super().__init__()
        self.__name = name

    @abstractmethod
    def dependencies(self) -> Set[str]:
        pass

    @abstractmethod
    def __call__(self, xs, args: ArgsDict):
        pass

    @abstractmethod
    def sample(self, xs, samples: SamplesDict):
        pass

    @property
    def name(self) -> str:
        return self.__name


class AdditionNode(ASTNode):
    def __init__(self, name: str, lhs: str, rhs: str):
        super().__init__(name)
        self.__lhs = lhs
        self.__rhs = rhs

    @property
    def lhs(self) -> str:
        return self.__lhs

    @property
    def rhs(self) -> str:
        return self.__rhs

    @cached_property
    def dependencies(self) -> Set[str]:
        return set([self.lhs, self.rhs])

    def __call__(self, xs, args: ArgsDict):
        lhs, lhs_constraint = args[self.lhs]
        rhs, rhs_constraint = args[self.rhs]
        res = lhs + rhs
        res_constraint = lhs_constraint + rhs_constraint
        args[self.name] = (res, res_constraint)

    def sample(self, xs, samples: SamplesDict):
        lhs = samples[self.lhs]
        rhs = samples[self.rhs]
        samples[self.name] = lhs + rhs

    def __repr__(self) -> str:
        return f'Addition(name: {self.name}, lhs: {self.lhs}, rhs: {self.rhs})'


class SubtractionNode(ASTNode):
    def __init__(self, name: str, minuend: str, subtrahend: str):
        super().__init__(name)
        self.__minuend = minuend
        self.__subtrahend = subtrahend

    @property
    def minuend(self) -> str:
        return self.__minuend

    @property
    def subtrahend(self) -> str:
        return self.__subtrahend

    @cached_property
    def dependencies(self) -> Set[str]:
        return set([self.minuend, self.subtrahend])

    def __call__(self, xs, args: ArgsDict):
        minuend, minuend_constraint = args[self.minuend]
        subtrahend, subtrahend_constraint = args[self.subtrahend]
        res = minuend - subtrahend
        res_constraint = minuend_constraint + subtrahend_constraint
        args[self.name] = (res, res_constraint)

    def sample(self, xs, samples: SamplesDict):
        minuend = samples[self.minuend]
        subtrahend = samples[self.subtrahend]
        samples[self.name] = minuend - subtrahend

    def __repr__(self) -> str:
        return f'Subtraction(name: {self.name}, minuend: {self.minuend}, subtrahend: {self.subtrahend})'


class MultiplicationNode(ASTNode):
    def __init__(self, name: str, lhs: str, rhs: str):
        super().__init__(name)
        self.__lhs = lhs
        self.__rhs = rhs

    @property
    def lhs(self) -> str:
        return self.__lhs

    @property
    def rhs(self) -> str:
        return self.__rhs

    @cached_property
    def dependencies(self) -> Set[str]:
        return set([self.lhs, self.rhs])

    def __call__(self, xs, args: ArgsDict):
        lhs, lhs_constraint = args[self.lhs]
        rhs, rhs_constraint = args[self.rhs]
        res = lhs * rhs
        res_constraint = lhs_constraint + rhs_constraint
        args[self.name] = (res, res_constraint)

    def sample(self, xs, samples: SamplesDict):
        lhs = samples[self.lhs]
        rhs = samples[self.rhs]
        samples[self.name] = lhs * rhs

    def __repr__(self) -> str:
        return f'Multiplication(name: {self.name}, lhs: {self.lhs}, rhs: {self.rhs})'


class DivisionNode(ASTNode):
    def __init__(self, name: str, dividend: str, divisor: str):
        super().__init__(name)
        self.__dividend = dividend
        self.__divisor = divisor

    @property
    def dividend(self) -> str:
        return self.__dividend

    @property
    def divisor(self) -> str:
        return self.__divisor

    @cached_property
    def dependencies(self) -> Set[str]:
        return set([self.dividend, self.divisor])

    def __call__(self, xs, args: ArgsDict):
        dividend, dividend_constraint = args[self.dividend]
        divisor, divisor_constraint = args[self.divisor]
        res = dividend / divisor
        res_constraint = dividend_constraint + divisor_constraint
        args[self.name] = (res, res_constraint)

    def sample(self, xs, samples: SamplesDict):
        dividend = samples[self.dividend]
        divisor = samples[self.divisor]
        samples[self.name] = dividend / divisor

    def __repr__(self) -> str:
        return f'Division(name: {self.name}, dividend: {self.dividend}, divisor: {self.divisor})'


class SingleNodeAST(ASTNode):
    def __init__(self, name: str, arg: str):
        super().__init__(name)
        self.__arg = arg

    @property
    def arg(self):
        return self.__arg

    @cached_property
    def dependencies(self) -> Set[str]:
        return set([self.arg])


class SingleNodeIdentity(SingleNodeAST):
    def __init__(self, name: str, arg: str):
        super().__init__(name, arg)

    def __call__(self, xs, args: ArgsDict):
        args[self.name] = args[self.arg]

    def sample(self, xs, samples: SamplesDict):
        samples[self.name] = samples[self.arg]

    def __repr__(self) -> str:
        return f'SingleNodeIdentity(name: {self.name}, arg: {self.arg})'


class AdditiveInverseNode(SingleNodeAST):
    def __init__(self, name: str, arg: str):
        super().__init__(name, arg)

    def __call__(self, xs, args: ArgsDict):
        arg, arg_constraint = args[self.arg]
        args[self.name] = (-arg, arg_constraint)

    def sample(self, xs, samples: SamplesDict):
        arg = samples[self.arg]
        samples[self.name] = -arg

    def __repr__(self) -> str:
        return f'AdditiveInverse(name: {self.name}, arg: {self.arg})'


class ExponentNode(SingleNodeAST):
    def __init__(self, name: str, arg: str):
        super().__init__(name, arg)

    def __call__(self, xs, args: ArgsDict):
        arg, arg_constraint = args[self.arg]
        args[self.name] = (arg.exp(), arg_constraint)

    def sample(self, xs, samples: SamplesDict):
        arg = samples[self.arg]
        samples[self.name] = arg.exp()

    def __repr__(self) -> str:
        return f'Exponent(name: {self.name}, arg: {self.arg})'


class LogNode(SingleNodeAST):
    def __init__(self, name: str, arg: str):
        super().__init__(name, arg)

    def __call__(self, xs, args: ArgsDict):
        arg, arg_constraint = args[self.arg]
        args[self.name] = (arg.log(), arg_constraint)

    def sample(self, xs, samples: SamplesDict):
        arg = samples[self.arg]
        samples[self.name] = arg.log()

    def __repr__(self) -> str:
        return f'Log(name: {self.name}, arg: {self.arg})'
