import torch

from src.models.branch import (
    BranchOutput,
    MalwareImageBranch,
)
from src.training.lightning_classifier import (
    LightningMalwareClassifier,
)


class DummyBranch(MalwareImageBranch):

    def __init__(
        self,
        num_classes: int = 51,
    ):
        super().__init__(
            num_classes=num_classes
        )

        self.pool = torch.nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.classifier = torch.nn.Linear(
            1,
            num_classes,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> BranchOutput:

        embedding = self.pool(x).flatten(1)

        logits = self.classifier(
            embedding
        )

        return BranchOutput(
            logits=logits,
            embedding=embedding,
        )


def test_lightning_forward():

    branch = DummyBranch()

    model = LightningMalwareClassifier(
        model=branch
    )

    x = torch.randn(
        4,
        1,
        64,
        64,
    )

    output = model(x)

    assert output.logits.shape == (
        4,
        51,
    )

    assert output.embedding.shape == (
        4,
        1,
    )


def test_optimizer_is_adam():

    branch = DummyBranch()

    model = LightningMalwareClassifier(
        model=branch
    )

    optimizer = model.configure_optimizers()

    assert isinstance(
        optimizer,
        torch.optim.Adam,
    )