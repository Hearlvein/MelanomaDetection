"""Prediction, threshold selection, and one-shot test evaluation for deep models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import config
from ..training import evaluate_at_threshold, threshold_bundle


def predict_scores(model: Any, dataset: Any) -> np.ndarray:
    """Return the positive-class score for every example in a batched dataset."""
    scores = model.predict(dataset, verbose=0)
    return np.asarray(scores, dtype=float).reshape(-1)


def select_thresholds(
    y_true: np.ndarray | list[int],
    scores: np.ndarray | list[float],
    *,
    min_sensitivity: float = 0.90,
) -> dict[str, float | None]:
    """Select the F2 threshold and a secondary high-sensitivity threshold.

    The caller must pass validation-only scores.
    """
    return threshold_bundle(y_true, scores, min_sensitivity=min_sensitivity)


def evaluate_frame_scores(
    frame: pd.DataFrame,
    scores: np.ndarray | list[float],
    threshold: float,
    *,
    bootstrap_reps: int = 2_000,
    seed: int = config.SEED,
) -> dict[str, Any]:
    """Evaluate scores against a frame's target using lesion-clustered CIs."""
    if len(frame) != len(scores):
        raise ValueError(
            f"frame has {len(frame)} rows but {len(scores)} scores were supplied"
        )
    return evaluate_at_threshold(
        frame["target"].to_numpy(),
        np.asarray(scores, dtype=float),
        frame["lesion_id"].to_numpy(),
        threshold=threshold,
        bootstrap_reps=bootstrap_reps,
        seed=seed,
    )


def deep_test_paths(
    target: str,
    model_name: str,
    *,
    metrics_path: str | Path | None = None,
    markdown_path: str | Path | None = None,
) -> tuple[Path, Path]:
    default_metrics = config.METRICS_DIR / f"deep_final_test_{target}_{model_name}.json"
    default_markdown = config.METRICS_DIR / f"deep_final_test_{target}_{model_name}.md"
    return (
        Path(metrics_path) if metrics_path is not None else default_metrics,
        Path(markdown_path) if markdown_path is not None else default_markdown,
    )


def assert_deep_test_not_yet_evaluated(
    target: str,
    model_name: str,
    *,
    force: bool = False,
    metrics_path: str | Path | None = None,
) -> Path:
    """Refuse to overwrite an existing deep test report unless explicitly forced."""
    path, _ = deep_test_paths(target, model_name, metrics_path=metrics_path)
    if path.exists() and not force:
        raise FileExistsError(
            f"deep test report already exists: {path}. "
            "The test split must be evaluated once. Pass --force only if you "
            "intentionally need to overwrite the frozen result."
        )
    return path
