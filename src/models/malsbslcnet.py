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
    MalSBSLCNet architecture reproduced from Zhang et al.

    The feature extractor and classifier structure follow Fig. 4
    and Appendix A.

    The final output dimension is adapted to the project's
    training-derived family label space.
    """

    def __init__(
        self,
        num_classes: int,
    ) -> None:
        super().__init__(
            num_classes=num_classes
        )

        self.features = nn.Sequential(
            # Conv Stem
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

            # Six BaseBlocks from Fig. 4 / Appendix A
            BaseBlock(
                in_channels=16,
                out_channels=32,
                stride=2,
            ),
            BaseBlock(
                in_channels=32,
                out_channels=32,
                stride=1,
            ),
            BaseBlock(
                in_channels=32,
                out_channels=64,
                stride=2,
            ),
            BaseBlock(
                in_channels=64,
                out_channels=64,
                stride=1,
            ),
            BaseBlock(
                in_channels=64,
                out_channels=128,
                stride=2,
            ),
            BaseBlock(
                in_channels=128,
                out_channels=128,
                stride=1,
            ),
        )

        # Appendix A:
        # 128 x 8 x 8 -> 128 x 4 x 4
        self.pool = nn.AdaptiveAvgPool2d(
            (4, 4)
        )

        self.flatten = nn.Flatten()

        # 128 * 4 * 4 = 2048
        self.embedding_dim = 2048

        self.classifier_bn = nn.BatchNorm1d(
            self.embedding_dim
        )

        self.dropout = nn.Dropout(
            p=0.4
        )

        self.classifier = nn.Linear(
            self.embedding_dim,
            num_classes,
        )

    def forward_features(
        self,
        x: Tensor,
    ) -> Tensor:
        return self.features(x)

    def forward(
        self,
        x: Tensor,
    ) -> BranchOutput:

        x = self.features(x)

        x = self.pool(x)

        x = self.flatten(x)

        # Deterministic classifier feature before dropout.
        embedding = self.classifier_bn(x)

        logits = self.classifier(
            self.dropout(embedding)
        )

        return BranchOutput(
            logits=logits,
            embedding=embedding,
        )