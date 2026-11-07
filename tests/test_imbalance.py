from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from melanoma_detection import config
from melanoma_detection.imbalance import (
    build_imbalance_pipeline,
    class_weight_for,
    make_sampler,
    strategies_for_model,
)


def _toy_frame(n_rows: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data: dict[str, object] = {}
    for offset, column in enumerate(config.NUMERIC_FEATURES):
        data[column] = rng.normal(size=n_rows) + offset
    data["sex"] = rng.choice(["male", "female"], size=n_rows)
    data["localization"] = rng.choice(["back", "trunk"], size=n_rows)
    return pd.DataFrame(data)


def _imbalanced_labels(n_rows: int = 40) -> np.ndarray:
    # 30 negatives, 10 positives.
    return np.array([0] * 30 + [1] * 10)


def test_class_weight_for_maps_strategies() -> None:
    assert class_weight_for("none", "logistic_regression") is None
    assert class_weight_for("balanced", "logistic_regression") == "balanced"
    assert class_weight_for("balanced", "random_forest") == "balanced"
    assert class_weight_for("balanced_subsample", "random_forest") == "balanced_subsample"
    assert class_weight_for("random_over", "random_forest") is None
    assert class_weight_for("smote", "hist_gradient_boosting") is None


def test_balanced_subsample_is_rejected_for_non_tree_models() -> None:
    with pytest.raises(ValueError, match="balanced_subsample"):
        class_weight_for("balanced_subsample", "logistic_regression")


def test_unknown_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown imbalance strategy"):
        class_weight_for("magic", "random_forest")
    with pytest.raises(ValueError, match="unknown imbalance strategy"):
        make_sampler("magic")


def test_strategies_for_model_excludes_balanced_subsample_for_linear_models() -> None:
    linear = strategies_for_model("logistic_regression")
    tree = strategies_for_model("random_forest")

    assert "balanced_subsample" not in linear
    assert "balanced_subsample" in tree
    assert "smote" in linear


def test_make_sampler_returns_expected_sampler_types() -> None:
    assert make_sampler("none") is None
    assert make_sampler("balanced") is None
    assert isinstance(make_sampler("random_over"), RandomOverSampler)
    assert isinstance(make_sampler("random_under"), RandomUnderSampler)
    assert isinstance(make_sampler("smote"), SMOTE)


def test_weight_strategy_builds_plain_sklearn_pipeline() -> None:
    pipeline = build_imbalance_pipeline(
        LogisticRegression(max_iter=2000, class_weight="balanced"),
        strategy="balanced",
    )

    assert isinstance(pipeline, Pipeline)
    assert "sampler" not in pipeline.named_steps


def test_resampling_strategy_puts_sampler_before_classifier() -> None:
    pipeline = build_imbalance_pipeline(
        LogisticRegression(max_iter=2000),
        strategy="random_over",
    )

    assert isinstance(pipeline, ImbalancedPipeline)
    step_names = list(pipeline.named_steps)
    assert step_names.index("sampler") < step_names.index("classifier")


def test_resampling_only_affects_training_fit_not_prediction_rows() -> None:
    train = _toy_frame()
    y = _imbalanced_labels()
    pipeline = build_imbalance_pipeline(
        LogisticRegression(max_iter=2000),
        strategy="random_over",
        k=None,
    )

    pipeline.fit(train, y)
    predictions = pipeline.predict_proba(train)

    # Prediction returns one row per input row; the sampler must not expand it.
    assert predictions.shape[0] == len(train)
    assert pipeline.named_steps["sampler"].sampling_strategy_ is not None


def test_preprocessing_statistics_come_from_the_original_training_fold() -> None:
    train = _toy_frame()
    train.loc[0, "age"] = np.nan
    y = _imbalanced_labels()
    pipeline = build_imbalance_pipeline(
        LogisticRegression(max_iter=2000),
        strategy="smote",
        k=None,
    )

    pipeline.fit(train, y)

    imputer = pipeline.named_steps["preprocessor"].named_transformers_["num"].named_steps[
        "imputer"
    ]
    assert imputer.statistics_[0] == pytest.approx(train["age"].median())
