from typing import Any

import lightning as L
import torch
from torch import Tensor
from torch import nn

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
)

from src.models.branch import (
    MalwareImageBranch,
)


class LightningMalwareClassifier(L.LightningModule):
    """
    Shared Lightning training wrapper for one malware-image branch.

    The wrapped model remains a normal PyTorch MalwareImageBranch.

    This module handles:
    - cross-entropy loss;
    - Adam optimization;
    - development metrics;
    - training/validation logging.

    Chronological split selection is handled outside this class.
    """

    def __init__(
        self,
        model: MalwareImageBranch,
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__()

        self.model = model
        self.learning_rate = learning_rate

        self.criterion = nn.CrossEntropyLoss()

        num_classes = model.num_classes

        self.train_accuracy = MulticlassAccuracy(
            num_classes=num_classes,
        )

        self.val_accuracy = MulticlassAccuracy(
            num_classes=num_classes,
        )

        self.val_macro_f1 = MulticlassF1Score(
            num_classes=num_classes,
            average="macro",
        )

        self.save_hyperparameters(
            ignore=["model"]
        )

    def forward(
        self,
        x: Tensor,
    ):
        return self.model(x)

    def training_step(
        self,
        batch: tuple[Tensor, Tensor],
        batch_idx: int,
    ) -> Tensor:

        x, y = batch

        output = self.model(x)

        loss = self.criterion(
            output.logits,
            y,
        )

        predictions = output.logits.argmax(
            dim=1
        )

        self.train_accuracy(
            predictions,
            y,
        )

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.log(
            "train_accuracy",
            self.train_accuracy,
            on_step=False,
            on_epoch=True,
        )

        return loss

    def validation_step(
        self,
        batch: tuple[Tensor, Tensor],
        batch_idx: int,
    ) -> None:

        x, y = batch

        output = self.model(x)

        loss = self.criterion(
            output.logits,
            y,
        )

        predictions = output.logits.argmax(
            dim=1
        )

        self.val_accuracy(
            predictions,
            y,
        )

        self.val_macro_f1(
            predictions,
            y,
        )

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.log(
            "val_accuracy",
            self.val_accuracy,
            on_step=False,
            on_epoch=True,
        )

        self.log(
            "val_macro_f1",
            self.val_macro_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def configure_optimizers(
        self,
    ) -> torch.optim.Optimizer:

        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )