from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from melanoma_detection.evaluate import (
    binary_metrics,
    cluster_bootstrap_metric_cis,
    select_f2_threshold,
    select_sensitivity_threshold,
)


def test_binary_metrics_use_class_one_as_positive() -> None:
    y_true = [1, 1, 1, 0, 0, 0]
    y_score = [0.9, 0.6, 0.4, 0.8, 0.3, 0.1]

    metrics = binary_metrics(y_true, y_score, threshold=0.5)

    assert metrics["confusion_matrix"] == {"tn": 2, "fp": 1, "fn": 1, "tp": 2}
    assert metrics["sensitivity"] == pytest.approx(2 / 3)
    assert metrics["specificity"] == pytest.approx(2 / 3)
    assert metrics["ppv"] == pytest.approx(2 / 3)
    assert metrics["npv"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["f2"] == pytest.approx(2 / 3)
    assert metrics["balanced_accuracy"] == pytest.approx(2 / 3)
    assert metrics["average_precision"] == pytest.approx(
        average_precision_score(y_true, y_score)
    )
    assert metrics["roc_auc"] == pytest.approx(roc_auc_score(y_true, y_score))
    assert metrics["brier_score"] == pytest.approx(
        float(np.mean((np.asarray(y_score) - np.asarray(y_true)) ** 2))
    )


def test_threshold_changes_the_confusion_matrix() -> None:
    y_true = [1, 1, 0, 0]
    y_score = [0.9, 0.4, 0.6, 0.1]

    low = binary_metrics(y_true, y_score, threshold=0.3)
    high = binary_metrics(y_true, y_score, threshold=0.7)

    assert low["confusion_matrix"]["tp"] == 2
    assert low["confusion_matrix"]["fp"] == 1
    assert high["confusion_matrix"]["tp"] == 1
    assert high["confusion_matrix"]["fp"] == 0


def test_f2_threshold_selection_is_deterministic() -> None:
    y_true = [0, 0, 0, 1, 1, 1]
    y_score = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]

    first = select_f2_threshold(y_true, y_score)
    second = select_f2_threshold(y_true, y_score)

    assert first == second
    assert first == pytest.approx(0.7)
    selected = binary_metrics(y_true, y_score, threshold=first)
    assert selected["sensitivity"] == pytest.approx(1.0)
    assert selected["ppv"] == pytest.approx(1.0)


def test_sensitivity_threshold_respects_the_constraint() -> None:
    y_true = [0, 0, 0, 1, 1, 1]
    y_score = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]

    threshold = select_sensitivity_threshold(y_true, y_score, min_sensitivity=0.90)

    assert threshold is not None
    metrics = binary_metrics(y_true, y_score, threshold=threshold)
    assert metrics["sensitivity"] >= 0.90


def test_invalid_sensitivity_target_is_rejected() -> None:
    with pytest.raises(ValueError, match="min_sensitivity"):
        select_sensitivity_threshold([0, 1], [0.2, 0.8], min_sensitivity=1.1)


def test_cluster_bootstrap_is_reproducible_and_returns_intervals() -> None:
    rng = np.random.default_rng(0)
    groups = np.repeat([f"lesion_{index}" for index in range(20)], 3)
    y_true = rng.integers(0, 2, size=len(groups))
    y_score = np.clip(rng.normal(0.5, 0.2, size=len(groups)), 0, 1)

    first = cluster_bootstrap_metric_cis(
        y_true, y_score, groups, threshold=0.5, n_boot=100, seed=7
    )
    second = cluster_bootstrap_metric_cis(
        y_true, y_score, groups, threshold=0.5, n_boot=100, seed=7
    )

    assert first == second
    assert "sensitivity" in first
    assert first["sensitivity"]["lower"] <= first["sensitivity"]["upper"]
    assert 0 <= first["sensitivity"]["lower"] <= 1
    assert 0 <= first["sensitivity"]["upper"] <= 1


def test_cluster_bootstrap_requires_matching_groups() -> None:
    with pytest.raises(ValueError, match="same length"):
        cluster_bootstrap_metric_cis(
            [0, 1, 0],
            [0.1, 0.8, 0.4],
            ["a", "b"],
            threshold=0.5,
            n_boot=10,
        )


def test_binary_metrics_rejects_non_binary_labels() -> None:
    with pytest.raises(ValueError, match="only 0 and 1"):
        binary_metrics([0, 1, 2], [0.1, 0.5, 0.9], threshold=0.5)
