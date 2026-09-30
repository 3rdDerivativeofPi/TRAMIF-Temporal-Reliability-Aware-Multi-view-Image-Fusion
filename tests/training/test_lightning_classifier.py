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


def test_training_step_backward():
    """
    Verify one forward + backward pass produces gradients.

    This test exercises the full training_step path:
    model forward -> cross-entropy loss -> backward pass.
    """
    torch.manual_seed(42)

    branch = DummyBranch(
        num_classes=51
    )

    model = LightningMalwareClassifier(
        model=branch
    )

    x = torch.randn(4, 1, 64, 64)
    y = torch.randint(0, 51, (4,))

    model.train()

    optimizer = model.configure_optimizers()

    loss = model.training_step((x, y), batch_idx=0)

    assert loss.requires_grad is True

    loss.backward()

    # At least one parameter should have a gradient
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
        if p.requires_grad
    )
    assert has_grad, "No parameter received a gradient after backward()"