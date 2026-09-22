import torch
from torch import Tensor
from torch import nn

from src.models.branch import (
    BranchOutput,
    MalwareImageBranch,
)
from src.models.malsbslcnet_blocks import (
    BaseBlock,
)


class MalSBSLCNet(MalwareImageBranch):
    """
    Lightweight CNN for 64x64 SBSMI inputs.

    Backbone structure follows Fig. 4 of Zhang et al.

    Note:
    exact AdaptiveAvgPool output dimensions remain to be
    verified against Appendix A.
    """

    def __init__(
        self,
        num_classes: int,
    ) -> None:
        super().__init__(
            num_classes=num_classes
        )

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            BaseBlock(
                16,
                32,
                stride=2,
            ),

            BaseBlock(
                32,
                32,
                stride=1,
            ),

            BaseBlock(
                32,
                64,
                stride=2,
            ),

            BaseBlock(
                64,
                64,
                stride=1,
            ),

            BaseBlock(
                64,
                128,
                stride=2,
            ),

            BaseBlock(
                128,
                128,
                stride=1,
            ),
        )