from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from melanoma_detection import config
from melanoma_detection.model import (
    DEFAULT_STRATEGIES,
    MODEL_ORDER,
    baseline_specs,
    build_baseline_pipeline,
    build_candidate_pipeline,
    build_imbalance_pipeline,
    build_pipeline,
    build_preprocessor,
    build_selector,
    candidate_specs,
    class_weight_for,
    make_sampler,
    selected_feature_count,
    strategies_for_model,
)


def _toy_frame(n_rows: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data: dict[str, object] = {}
    for offset, column in enumerate(config.NUMERIC_FEATURES):
        data[column] = rng.normal(size=n_rows) + offset
    data["sex"] = rng.choice(["male", "female"], size=n_rows)
    data["localization"] = rng.choice(["back", "trunk"], size=n_rows)
    return pd.DataFrame(data)


def _labels(n_rows: int = 60) -> np.ndarray:
    return np.array([1] * (n_rows // 4) + [0] * (n_rows - n_rows // 4))


def test_preprocessor_statistics_are_learned_from_train_only() -> None:
    train = pd.DataFrame(
        {
            "age": [10.0, 20.0, 30.0],
            "sex": ["male", "female", "male"],
            "localization": ["back", "trunk", "back"],
        }
    )
    validation = pd.DataFrame(
        {
            "age": [np.nan],
            "sex": ["female"],
            "localization": ["trunk"],
        }
    )
    preprocessor = build_preprocessor(
        numeric_features=("age",),
        categorical_features=("sex", "localization"),
        scale_numeric=False,
    )
    preprocessor.fit(train)

    transformed = preprocessor.transform(validation)

    assert transformed[0, 0] == pytest.approx(20.0)


def test_selector_keeps_the_requested_number_of_features() -> None:
    frame = _toy_frame()
    pipeline = build_pipeline(LogisticRegression(), k=5)
    pipeline.fit(frame, _labels())

    assert selected_feature_count(pipeline) == 5


def test_build_pipeline_requires_at_least_one_feature() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_pipeline(
            LogisticRegression(), numeric_features=(), categorical_features=()
        )


def test_class_weight_mapping() -> None:
    assert class_weight_for("none", "logistic_regression") is None
    assert class_weight_for("balanced", "logistic_regression") == "balanced"
    assert class_weight_for("smote", "random_forest") is None
    assert class_weight_for("balanced_subsample", "random_forest") == "balanced_subsample"

    with pytest.raises(ValueError, match="balanced_subsample"):
        class_weight_for("balanced_subsample", "logistic_regression")
    with pytest.raises(ValueError, match="unknown imbalance"):
        class_weight_for("mystery", "extra_trees")


def test_make_sampler_and_strategy_availability() -> None:
    assert make_sampler("none") is None
    assert make_sampler("balanced") is None
    assert type(make_sampler("smote")).__name__ == "SMOTE"

    assert strategies_for_model("random_forest") == (
        "none",
        "balanced",
        "balanced_subsample",
        "random_over",
        "random_under",
        "smote",
    )
    assert "balanced_subsample" not in strategies_for_model("logistic_regression")


def test_imbalance_pipeline_resamples_only_during_fit() -> None:
    frame = _toy_frame()
    labels = _labels()
    pipeline = build_imbalance_pipeline(
        LogisticRegression(max_iter=5_000), strategy="smote", k=5
    )

    assert "sampler" in pipeline.named_steps
    pipeline.fit(frame, labels)
    scores = pipeline.predict_proba(frame.iloc[[0, 1, 2]])

    assert scores.shape == (3, 2)


def test_candidate_specs_cover_the_four_model_families() -> None:
    specs = candidate_specs()
    assert [spec["name"] for spec in specs] == list(MODEL_ORDER)
    assert all("param_distributions" in spec for spec in specs)

    for spec in specs:
        pipeline = build_candidate_pipeline(spec, "balanced")
        assert "preprocessor" in pipeline.named_steps
        assert "selector" in pipeline.named_steps
        assert "classifier" in pipeline.named_steps


def test_baseline_specs_include_the_expected_subsets() -> None:
    names = [spec["name"] for spec in baseline_specs()]

    assert names == [
        "majority",
        "random",
        "age_only",
        "metadata_only",
        "colour_only",
        "shape_only",
        "all_features",
    ]
    assert DEFAULT_STRATEGIES == ("none", "balanced", "smote")


def test_age_only_baseline_uses_only_age() -> None:
    frame = _toy_frame()
    spec = next(spec for spec in baseline_specs() if spec["name"] == "age_only")
    pipeline = build_baseline_pipeline(spec)
    pipeline.fit(frame, _labels())

    assert pipeline.named_steps["preprocessor"].transformers[0][2] == ["age"]
    assert pipeline.predict_proba(frame.iloc[[0]]).shape == (1, 2)
