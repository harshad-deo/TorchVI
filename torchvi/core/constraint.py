import torch


class Constraint:
    def __init__(self, constraint_dict: dict):
        self.__constraint_dict = constraint_dict

    @staticmethod
    def new(name, tensor):
        if not isinstance(name, str):
            raise TypeError(f'name must be a string. Got: {type(name)}')
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f'tensor must be a torch Tensor. Got: {type(tensor)}')
        constraint_dict = {name: tensor}
        return Constraint(constraint_dict)

    @property
    def constraint_dict(self):
        return self.__constraint_dict

    def __add__(self, that):
        if not isinstance(that, Constraint):
            raise TypeError(f'Can only add constraints to another constraint. Got: {type(that)}')
        new_constraint_dict = {**self.__constraint_dict, **that.__constraint_dict}
        return Constraint(new_constraint_dict)

    def add_tensor(self, that):
        if not isinstance(that, torch.Tensor):
            raise TypeError(f'Expected tensor, got: {type(that)}')
        for value in self.__constraint_dict.values():
            that += value
        return that

    def __repr__(self) -> str:
        return f'{self.constraint_dict}'
