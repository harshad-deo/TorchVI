from collections.abc import Iterable
import torch


def to_size(x):
    """
    Converts x to torch.Size
    """
    if isinstance(x, torch.Size):
        return list(x)
    if isinstance(x, int):
        return [x]
    if isinstance(x, Iterable) and all([isinstance(y, int) for y in x]):
        return x
    raise TypeError(f'Bad type for vtensor size: {type(x)}')
