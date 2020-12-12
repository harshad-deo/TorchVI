from collections.abc import Iterable
import torch


def to_size(x):
    """
    Converts x to torch.Size
    """
    if isinstance(x, torch.Size):
        return x
    if isinstance(x, int):
        return torch.Size([x])
    if isinstance(x, Iterable) and all([isinstance(y, int) for y in x]):
        return torch.Size(x)
    raise TypeError(f'Bad type for vtensor size: {type(x)}')


def check_numeric(x, desc):
    if isinstance(x, float) or isinstance(x, int):
        return
    raise TypeError(f'{desc}: is not numeric')


def to_numeric_tensor(x, desc) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, float) or isinstance(x, int):
        return torch.tensor(x)
    raise TypeError(f'{desc} must be either a numeric or tensor. Got: {type(x)}')
