import torch
from torch import nn

from src.models.utils import (
    count_trainable_parameters,
    model_size_mb,
)


def test_parameter_count():
    model = nn.Linear(
        in_features=10,
        out_features=5,
    )

    # weights: 10 * 5 = 50
    # bias: 5
    assert count_trainable_parameters(model) == 55


def test_model_size_is_positive():
    model = nn.Linear(
        in_features=10,
        out_features=5,
    )

    assert model_size_mb(model) > 0