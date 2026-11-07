"""Class-weight and resampling strategies.

Resampling is deliberately implemented with an imbalanced-learn pipeline. The
sampler is a pipeline step, so it is fitted only on the training fold and is
skipped entirely during ``predict``/``predict_proba``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.pipeline import Pipeline

from . import config
from .features import build_preprocessor, build_selector

CLASS_WEIGHT_STRATEGIES = ("none", "balanced", "balanced_subsample")
SAMPLER_STRATEGIES = ("random_over", "random_under", "smote")
ALL_STRATEGIES = CLASS_WEIGHT_STRATEGIES + SAMPLER_STRATEGIES

TREE_MODELS = ("random_forest", "extra_trees")


def class_weight_for(strategy: str, model_name: str) -> str | None:
    """Map a strategy name to a valid ``class_weight`` for the model."""
    if strategy == "none":
        return None
    if strategy == "balanced":
        return "balanced"
    if strategy == "balanced_subsample":
        if model_name not in TREE_MODELS:
            raise ValueError(
                f"balanced_subsample is not supported for {model_name!r}; "
                "use 'balanced' or a resampling strategy"
            )
        return "balanced_subsample"
    if strategy in SAMPLER_STRATEGIES:
        return None
    raise ValueError(f"unknown imbalance strategy: {strategy!r}")


def make_sampler(strategy: str, *, seed: int = config.SEED) -> Any | None:
    """Build a training-fold-only sampler, or return None for weight strategies."""
    if strategy in CLASS_WEIGHT_STRATEGIES:
        return None
    if strategy == "random_over":
        return RandomOverSampler(random_state=seed)
    if strategy == "random_under":
        return RandomUnderSampler(random_state=seed)
    if strategy == "smote":
        return SMOTE(random_state=seed)
    raise ValueError(f"unknown imbalance strategy: {strategy!r}")


def strategies_for_model(model_name: str) -> tuple[str, ...]:
    """Return the strategies that are valid for a model family."""
    if model_name in TREE_MODELS:
        return ALL_STRATEGIES
    return tuple(strategy for strategy in ALL_STRATEGIES if strategy != "balanced_subsample")


def build_imbalance_pipeline(
    classifier: Any,
    *,
    strategy: str,
    scale_numeric: bool = True,
    k: int | str | None = config.DEFAULT_SELECT_K,
    numeric_features: Sequence[str] | None = None,
    categorical_features: Sequence[str] | None = None,
    seed: int = config.SEED,
) -> Pipeline | ImbalancedPipeline:
    """Build preprocess -> select -> optional sampler -> classify.

    The sampler is placed after feature selection and before the classifier. It
    only runs during ``fit``; validation/test data are never resampled.
    """
    sampler = make_sampler(strategy, seed=seed)
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
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("classifier", classifier))

    if sampler is not None:
        return ImbalancedPipeline(steps)
    return Pipeline(steps)
