from dataclasses import dataclass

import torch
from torch import Tensor
from torch import nn


@dataclass
class BranchOutput:
    logits: Tensor
    embedding: Tensor


class MalwareImageBranch(nn.Module):
    """
    Base interface for one binary-image classification branch.

    Every view-specific model must return:

    - logits over the same known-family class ordering;
    - a compact embedding usable for historical drift analysis.
    """

    def __init__(
        self,
        num_classes: int,
    ) -> None:
        super().__init__()

        if num_classes <= 1:
            raise ValueError(
                "num_classes must be greater than 1"
            )

        self.num_classes = num_classes

    def forward(
        self,
        x: Tensor,
    ) -> BranchOutput:
        raise NotImplementedError