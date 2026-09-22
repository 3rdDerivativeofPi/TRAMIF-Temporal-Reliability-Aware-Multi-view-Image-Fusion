import pytest
import torch

from src.models.branch import (
    BranchOutput,
    MalwareImageBranch,
)


class DummyBranch(MalwareImageBranch):
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 16,
    ):
        super().__init__(num_classes)

        self.embedding_dim = embedding_dim

    def forward(
        self,
        x: torch.Tensor,
    ) -> BranchOutput:

        batch_size = x.shape[0]

        logits = torch.zeros(
            batch_size,
            self.num_classes,
        )

        embedding = torch.zeros(
            batch_size,
            self.embedding_dim,
        )

        return BranchOutput(
            logits=logits,
            embedding=embedding,
        )


def test_branch_rejects_invalid_class_count():
    with pytest.raises(ValueError):
        DummyBranch(num_classes=1)


def test_branch_output_shapes():
    model = DummyBranch(
        num_classes=51,
        embedding_dim=16,
    )

    x = torch.zeros(
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
        16,
    )