import torch
from torch import nn


def count_trainable_parameters(
    model: nn.Module,
) -> int:
    """Return the number of trainable model parameters."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def model_size_mb(
    model: nn.Module,
) -> float:
    """
    Approximate parameter storage assuming the current
    parameter dtypes.
    """

    total_bytes = sum(
        parameter.numel()
        * parameter.element_size()
        for parameter in model.parameters()
    )

    return total_bytes / (1024 ** 2)