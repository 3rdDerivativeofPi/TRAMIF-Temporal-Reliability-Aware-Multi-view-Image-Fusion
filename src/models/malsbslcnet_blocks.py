import torch
from torch import Tensor
from torch import nn


def channel_shuffle(
    x: Tensor,
    groups: int = 2,
) -> Tensor:
    """
    Shuffle feature channels between groups.

    MalSBSLCNet uses channel shuffle after each BaseBlock
    to permit information exchange between branches.
    """

    if x.ndim != 4:
        raise ValueError(
            "Expected a 4D tensor [B, C, H, W]"
        )

    batch_size, channels, height, width = x.shape

    if channels % groups != 0:
        raise ValueError(
            "Number of channels must be divisible by groups"
        )

    channels_per_group = channels // groups

    x = x.reshape(
        batch_size,
        groups,
        channels_per_group,
        height,
        width,
    )

    x = x.transpose(1, 2).contiguous()

    return x.reshape(
        batch_size,
        channels,
        height,
        width,
    )

class BaseBlock(nn.Module):
    """
    ShuffleNetV2-style BaseBlock used by MalSBSLCNet.

    Supported source configurations:
    - stride = 1: input/output channels are equal
    - stride = 2: spatial downsampling and channel expansion
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
    ) -> None:
        super().__init__()

        if stride not in (1, 2):
            raise ValueError(
                "stride must be 1 or 2"
            )

        if out_channels % 2 != 0:
            raise ValueError(
                "out_channels must be even"
            )

        self.stride = stride

        if stride == 1:
            if in_channels != out_channels:
                raise ValueError(
                    "stride=1 requires "
                    "in_channels == out_channels"
                )

            if in_channels % 2 != 0:
                raise ValueError(
                    "stride=1 requires an even "
                    "number of channels"
                )

            branch_channels = in_channels // 2

            self.branch1 = nn.Identity()

            self.branch2 = nn.Sequential(
                nn.Conv2d(
                    branch_channels,
                    branch_channels,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    branch_channels
                ),
                nn.ReLU(inplace=True),

                nn.Conv2d(
                    branch_channels,
                    branch_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=branch_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    branch_channels
                ),

                nn.Conv2d(
                    branch_channels,
                    branch_channels,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    branch_channels
                ),
                nn.ReLU(inplace=True),
            )

        else:
            branch_channels = out_channels // 2

            self.branch1 = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=in_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    in_channels
                ),

                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    branch_channels
                ),
                nn.ReLU(inplace=True),
            )

            self.branch2 = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    branch_channels
                ),
                nn.ReLU(inplace=True),

                nn.Conv2d(
                    branch_channels,
                    branch_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=branch_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    branch_channels
                ),

                nn.Conv2d(
                    branch_channels,
                    branch_channels,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    branch_channels
                ),
                nn.ReLU(inplace=True),
            )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:

        if self.stride == 1:
            x1, x2 = x.chunk(
                chunks=2,
                dim=1,
            )

            y1 = x1
            y2 = self.branch2(x2)

        else:
            y1 = self.branch1(x)
            y2 = self.branch2(x)

        output = torch.cat(
            (y1, y2),
            dim=1,
        )

        return channel_shuffle(
            output,
            groups=2,
        )