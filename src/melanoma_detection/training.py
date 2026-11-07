"""Shared helpers for imbalance experiments and model selection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict

from . import config
from .evaluate import (
    binary_metrics,
    cluster_bootstrap_metric_cis,
    select_f2_threshold,
    select_sensitivity_threshold,
)


def positive_scores(probabilities: np.ndarray) -> np.ndarray:
    """Extract the positive-class probability from a binary probability array."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim == 1:
        return probabilities
    if probabilities.shape[1] != 2:
        raise ValueError(
            f"expected binary predict_proba output with two columns, got {probabilities.shape}"
        )
    return probabilities[:, 1]


def oof_positive_scores(
    estimator: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    cv,
) -> np.ndarray:
    """Generate grouped out-of-fold positive-class scores inside the training set."""
    probabilities = cross_val_predict(
        estimator,
        X,
        y,
        groups=groups,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )
    return positive_scores(probabilities)


def threshold_bundle(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    min_sensitivity: float = 0.90,
) -> dict[str, float | None]:
    """Select the primary F2 threshold and the secondary high-sensitivity threshold."""
    return {
        "f2_threshold": float(select_f2_threshold(y_true, y_score)),
        "sensitivity_target": float(min_sensitivity),
        "sensitivity_threshold": select_sensitivity_threshold(
            y_true, y_score, min_sensitivity=min_sensitivity
        ),
    }


def evaluate_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    threshold: float,
    *,
    bootstrap_reps: int = 2_000,
    seed: int = config.SEED,
) -> dict[str, Any]:
    """Compute binary metrics and lesion-clustered confidence intervals."""
    metrics = binary_metrics(y_true, y_score, threshold=threshold)
    confidence_intervals = cluster_bootstrap_metric_cis(
        y_true,
        y_score,
        groups,
        threshold=threshold,
        n_boot=bootstrap_reps,
        seed=seed,
    )
    return {
        "metrics": metrics,
        "confidence_intervals": confidence_intervals,
        "confusion_matrix": metrics["confusion_matrix"],
    }


def split_counts(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Summarise rows, lesions, and positive counts for each split."""
    counts: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation", "test"):
        subset = frame.loc[frame["split"] == split]
        positive = int(subset["target"].sum())
        counts[split] = {
            "rows": int(len(subset)),
            "lesions": int(subset["lesion_id"].nunique()),
            "positive": positive,
            "negative": int(len(subset) - positive),
            "prevalence": float(positive / len(subset)) if len(subset) else float("nan"),
        }
    return counts


def assert_split_isolation(frame: pd.DataFrame) -> None:
    """Fail if any lesion appears in more than one split."""
    lesion_sets = {
        split: set(frame.loc[frame["split"] == split, "lesion_id"])
        for split in ("train", "validation", "test")
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        shared = lesion_sets[left] & lesion_sets[right]
        if shared:
            raise RuntimeError(
                f"{len(shared)} lesions overlap between {left} and {right}; "
                "rerun the Phase 1 split before evaluating."
            )
