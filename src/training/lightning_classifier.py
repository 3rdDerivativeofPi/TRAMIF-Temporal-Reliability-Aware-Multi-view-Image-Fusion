from typing import Any

import lightning as L
import torch
from torch import Tensor
from torch import nn

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
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
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()

        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

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

        self.test_accuracy = MulticlassAccuracy(
            num_classes=num_classes,
            average="micro",
        )

        self.test_macro_precision = MulticlassPrecision(
            num_classes=num_classes,
            average="macro",
            zero_division=0,
        )

        self.test_macro_recall = MulticlassRecall(
            num_classes=num_classes,
            average="macro",
            zero_division=0,
        )

        self.test_macro_f1 = MulticlassF1Score(
            num_classes=num_classes,
            average="macro",
            zero_division=0,
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

    def test_step(
        self,
        batch: tuple[Tensor, Tensor],
        batch_idx: int,
    ) -> None:
        x, y = batch

        output = self.model(x)

        loss = self.criterion(output.logits, y)
        predictions = output.logits.argmax(dim=1)

        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=y.numel(),
        )

        for name, metric in (
            ("test_accuracy", self.test_accuracy),
            ("test_macro_precision", self.test_macro_precision),
            ("test_macro_recall", self.test_macro_recall),
            ("test_macro_f1", self.test_macro_f1),
        ):
            metric.update(predictions, y)

            self.log(
                name,
                metric,
                on_step=False,
                on_epoch=True,
            )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )