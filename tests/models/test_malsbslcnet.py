import torch

from src.models.malsbslcnet import (
    MalSBSLCNet,
)


def test_feature_shape():
    model = MalSBSLCNet(
        num_classes=51
    )

    model.eval()

    x = torch.randn(
        2,
        1,
        64,
        64,
    )

    with torch.no_grad():
        features = model.forward_features(x)

    assert features.shape == (
        2,
        128,
        8,
        8,
    )


def test_pool_shape():
    model = MalSBSLCNet(
        num_classes=51
    )

    model.eval()

    x = torch.randn(
        2,
        1,
        64,
        64,
    )

    with torch.no_grad():
        features = model.forward_features(x)
        pooled = model.pool(features)

    assert pooled.shape == (
        2,
        128,
        4,
        4,
    )


def test_embedding_shape():
    model = MalSBSLCNet(
        num_classes=51
    )

    model.eval()

    x = torch.randn(
        2,
        1,
        64,
        64,
    )

    with torch.no_grad():
        output = model(x)

    assert output.embedding.shape == (
        2,
        2048,
    )


def test_logits_shape():
    model = MalSBSLCNet(
        num_classes=51
    )

    model.eval()

    x = torch.randn(
        2,
        1,
        64,
        64,
    )

    with torch.no_grad():
        output = model(x)

    assert output.logits.shape == (
        2,
        51,
    )


def test_source_dropout_rate():
    model = MalSBSLCNet(
        num_classes=51
    )

    assert model.dropout.p == 0.4


def test_classifier_dimensions():
    model = MalSBSLCNet(
        num_classes=51
    )

    assert model.classifier.in_features == 2048
    assert model.classifier.out_features == 51

def test_source_classifier_dimensions():
    model = MalSBSLCNet(
        num_classes=9
    )

    assert model.classifier.in_features == 2048
    assert model.classifier.out_features == 9

    classifier_params = sum(
        p.numel()
        for p in model.classifier.parameters()
    )

    assert classifier_params == 18441