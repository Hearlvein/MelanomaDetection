from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sklearn.pipeline import Pipeline

from . import config
from .features import build_preprocessor, build_selector


def build_pipeline(
    classifier: Any,
    *,
    scale_numeric: bool = True,
    k: int | str | None = config.DEFAULT_SELECT_K,
    numeric_features: Sequence[str] | None = None,
    categorical_features: Sequence[str] | None = None,
) -> Pipeline:
    """Build a single leak-free pipeline: preprocess -> select -> classify.

    Always pass raw feature columns to ``fit``/``cross_val_score``. Fitting the
    preprocessor or selector outside the pipeline is what caused the leakage in
    the legacy notebooks.

    ``numeric_features`` and ``categorical_features`` default to the full schema
    and can be overridden for baseline/ablation models.
    """
    steps: list[tuple[str, Any]] = [
        (
            "preprocessor",
            build_preprocessor(
                scale_numeric=scale_numeric,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
            ),
        )
    ]
    if k is not None:
        steps.append(("selector", build_selector(k)))
    steps.append(("classifier", classifier))
    return Pipeline(steps)


def selected_feature_count(pipeline: Pipeline) -> int | None:
    """Number of features kept by the selector, or ``None`` if no selector."""
    if "selector" not in pipeline.named_steps:
        return None
    selector = pipeline.named_steps["selector"]
    if not hasattr(selector, "get_support"):
        return None
    return int(selector.get_support().sum())
