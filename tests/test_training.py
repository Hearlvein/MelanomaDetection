from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from melanoma_detection.training import (
    assert_split_isolation,
    evaluate_at_threshold,
    positive_scores,
    split_counts,
    threshold_bundle,
)


def test_positive_scores_accepts_one_and_two_dimensional_input() -> None:
    assert np.array_equal(
        positive_scores(np.array([0.1, 0.9])),
        np.array([0.1, 0.9]),
    )
    assert np.array_equal(
        positive_scores(np.array([[0.9, 0.1], [0.2, 0.8]])),
        np.array([0.1, 0.8]),
    )


def test_positive_scores_rejects_non_binary_probability_output() -> None:
    with pytest.raises(ValueError, match="two columns"):
        positive_scores(np.zeros((3, 4)))


def test_threshold_bundle_selects_f2_and_sensitivity_thresholds() -> None:
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_score = np.array([0.05, 0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.95])

    thresholds = threshold_bundle(y_true, y_score, min_sensitivity=0.90)

    assert 0 <= thresholds["f2_threshold"] <= 1
    assert thresholds["sensitivity_target"] == pytest.approx(0.90)
    assert thresholds["sensitivity_threshold"] is not None
    assert thresholds["sensitivity_threshold"] <= thresholds["f2_threshold"]


def test_evaluate_at_threshold_returns_metrics_and_intervals() -> None:
    rng = np.random.default_rng(0)
    groups = np.repeat([f"lesion_{index}" for index in range(20)], 3)
    y_true = (rng.random(len(groups)) < 0.3).astype(int)
    y_score = np.clip(rng.normal(0.4, 0.25, size=len(groups)), 0, 1)

    evaluation = evaluate_at_threshold(
        y_true,
        y_score,
        groups,
        threshold=0.5,
        bootstrap_reps=100,
        seed=7,
    )

    assert evaluation["metrics"]["threshold"] == pytest.approx(0.5)
    assert evaluation["confusion_matrix"] == evaluation["metrics"]["confusion_matrix"]
    assert "sensitivity" in evaluation["confidence_intervals"]


def test_split_counts_summarises_each_split() -> None:
    frame = pd.DataFrame(
        {
            "split": ["train", "train", "train", "validation", "validation", "test"],
            "lesion_id": ["a", "a", "b", "c", "d", "e"],
            "target": [1, 0, 1, 1, 0, 0],
        }
    )

    counts = split_counts(frame)

    assert counts["train"]["rows"] == 3
    assert counts["train"]["lesions"] == 2
    assert counts["train"]["positive"] == 2
    assert counts["train"]["negative"] == 1
    assert counts["train"]["prevalence"] == pytest.approx(2 / 3)
    assert counts["validation"]["rows"] == 2
    assert counts["test"]["positive"] == 0


def test_assert_split_isolation_accepts_disjoint_lesions() -> None:
    frame = pd.DataFrame(
        {
            "split": ["train", "train", "validation", "test"],
            "lesion_id": ["a", "b", "c", "d"],
        }
    )

    assert_split_isolation(frame)


def test_assert_split_isolation_rejects_shared_lesions() -> None:
    frame = pd.DataFrame(
        {
            "split": ["train", "validation", "test"],
            "lesion_id": ["shared", "shared", "other"],
        }
    )

    with pytest.raises(RuntimeError, match="lesions overlap"):
        assert_split_isolation(frame)
