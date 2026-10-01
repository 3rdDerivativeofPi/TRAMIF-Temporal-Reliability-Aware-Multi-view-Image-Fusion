"""MalCSBSV reproduction settings from Section 5.1, Table 3."""

from src.models.malsbslcnet import MalSBSLCNet
from src.training.lightning_classifier import (
    LightningMalwareClassifier,
)


def make_malsbslcnet(
    *,
    num_classes: int,
) -> LightningMalwareClassifier:
    return LightningMalwareClassifier(
        model=MalSBSLCNet(
            num_classes=num_classes,
        ),
        learning_rate=1e-4,
        weight_decay=1e-6,
    )