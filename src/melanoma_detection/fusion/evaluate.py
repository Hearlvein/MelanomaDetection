"""One-shot test evaluation helpers for the late-fusion line.

The fusion model follows the same rules as the tabular Phase 5 line: the
threshold is frozen from grouped out-of-fold scores inside the training split,
validation is used for model selection, and the test split is evaluated once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import config
from ..training import evaluate_at_threshold
from .train import fusion_scores


def fusion_test_paths(
    target: str = config.TARGET_DEFINITION,
    *,
    metrics_path: str | Path | None = None,
    markdown_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Return the JSON and Markdown paths for the frozen fusion test report."""
    default_metrics = config.METRICS_DIR / f"fusion_final_test_{target}.json"
    default_markdown = config.METRICS_DIR / f"fusion_final_test_{target}.md"
    return (
        Path(metrics_path) if metrics_path is not None else default_metrics,
        Path(markdown_path) if markdown_path is not None else default_markdown,
    )


def assert_fusion_test_not_yet_evaluated(
    target: str = config.TARGET_DEFINITION,
    *,
    force: bool = False,
    metrics_path: str | Path | None = None,
) -> Path:
    """Refuse to overwrite an existing fusion test report unless forced."""
    path, _ = fusion_test_paths(target, metrics_path=metrics_path)
    if path.exists() and not force:
        raise FileExistsError(
            f"fusion test report already exists: {path}. "
            "The test split must be evaluated once. Pass --force only if you "
            "intentionally need to overwrite the frozen result."
        )
    return path


def evaluate_fusion_frame(
    pipeline: Any,
    frame: pd.DataFrame,
    feature_columns: list[str],
    threshold: float,
    *,
    bootstrap_reps: int = 2_000,
    seed: int = config.SEED,
) -> tuple[dict[str, Any], np.ndarray]:
    """Score a frame and compute metrics with lesion-clustered confidence intervals."""
    if frame.empty:
        raise ValueError("cannot evaluate an empty frame")
    scores = fusion_scores(pipeline, frame, feature_columns)
    if config.GROUP_COLUMN not in frame.columns:
        raise ValueError(f"frame is missing the group column {config.GROUP_COLUMN!r}")
    evaluation = evaluate_at_threshold(
        frame["target"].to_numpy(),
        scores,
        frame[config.GROUP_COLUMN].to_numpy(),
        threshold=threshold,
        bootstrap_reps=bootstrap_reps,
        seed=seed,
    )
    return evaluation, scores
