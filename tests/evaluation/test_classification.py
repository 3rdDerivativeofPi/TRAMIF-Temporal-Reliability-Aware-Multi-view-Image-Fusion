import pytest

from src.evaluation.classification import (
    compute_classification_metrics,
)


def test_perfect_predictions_have_perfect_metrics():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 2]

    metrics = compute_classification_metrics(
        y_true,
        y_pred,
        labels=[0, 1, 2],
    )

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["macro_precision"] == pytest.approx(1.0)
    assert metrics["macro_recall"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(1.0)


def test_known_binary_example_matches_hand_calculation():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]

    metrics = compute_classification_metrics(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["macro_precision"] == pytest.approx(5 / 6)
    assert metrics["macro_recall"] == pytest.approx(0.75)
    assert metrics["macro_f1"] == pytest.approx(11 / 15)


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="same number"):
        compute_classification_metrics(
            [0, 1],
            [0],
        )


def test_empty_inputs_are_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        compute_classification_metrics(
            [],
            [],
        )
