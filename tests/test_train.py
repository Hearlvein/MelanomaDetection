from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from melanoma_detection import config
from melanoma_detection.train import (
    CANDIDATE_ORDER,
    artifact_path_for,
    assert_test_not_yet_evaluated,
    build_candidate_pipeline,
    candidate_specs,
    evaluate_tuned_candidate,
    final_test_paths,
    frozen_pointer_path,
    load_artifact,
    public_candidate_result,
    resolve_artifact_path,
    save_artifact,
    select_final_candidate,
    tune_candidate,
)


def _toy_frame(n_lesions: int, images_per_lesion: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n_rows = n_lesions * images_per_lesion
    data: dict[str, object] = {
        "image_id": [f"img_{index}" for index in range(n_rows)],
        "lesion_id": np.repeat(
            [f"lesion_{index}" for index in range(n_lesions)], images_per_lesion
        ),
    }
    for offset, column in enumerate(config.NUMERIC_FEATURES):
        data[column] = rng.normal(size=n_rows) + offset
    data["sex"] = rng.choice(["male", "female"], size=n_rows)
    data["localization"] = rng.choice(["back", "trunk"], size=n_rows)
    return pd.DataFrame(data)


def _toy_spec() -> dict:
    return {
        "name": "logistic_regression",
        "label": "Logistic Regression",
        "estimator_factory": lambda class_weight: LogisticRegression(
            max_iter=2_000, class_weight=class_weight, random_state=config.SEED
        ),
        "scale_numeric": True,
        "default_strategy": "none",
        "strategy": "none",
        "param_distributions": {
            "classifier__C": [0.1, 1.0, 10.0],
            "selector__k": [3, 5],
        },
    }


def _toy_split() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    train = _toy_frame(n_lesions=20)
    # The first 10 lesions are positive, the rest negative.
    y_train = np.repeat([1] * 10 + [0] * 10, 3)
    groups_train = train["lesion_id"].to_numpy()

    validation = _toy_frame(n_lesions=10)
    validation["lesion_id"] = np.repeat(
        [f"val_lesion_{index}" for index in range(10)], 3
    )
    y_validation = np.repeat([1] * 5 + [0] * 5, 3)
    groups_validation = validation["lesion_id"].to_numpy()
    return train, y_train, groups_train, validation, y_validation, groups_validation


def test_candidate_specs_respect_strategy_overrides() -> None:
    specs = candidate_specs({"logistic_regression": "random_over"})
    by_name = {spec["name"]: spec for spec in specs}

    assert tuple(by_name) == CANDIDATE_ORDER
    assert by_name["logistic_regression"]["strategy"] == "random_over"
    assert by_name["random_forest"]["strategy"] == "balanced"
    assert all("selector__k" in spec["param_distributions"] for spec in specs)


def test_build_candidate_pipeline_uses_the_requested_strategy() -> None:
    spec = _toy_spec()
    pipeline = build_candidate_pipeline(spec, "random_over")

    assert "sampler" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


def test_tuning_and_freezing_produces_consistent_threshold() -> None:
    from sklearn.model_selection import StratifiedGroupKFold

    train, y_train, groups_train, validation, y_validation, groups_validation = _toy_split()
    cv = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=config.SEED)
    spec = _toy_spec()
    feature_columns = list(config.ALL_FEATURES)

    search = tune_candidate(
        spec,
        "none",
        X=train[feature_columns],
        y=y_train,
        groups=groups_train,
        cv=cv,
        n_iter=2,
        seed=config.SEED,
    )
    result = evaluate_tuned_candidate(
        search,
        spec,
        "none",
        X_train=train[feature_columns],
        y_train=y_train,
        groups_train=groups_train,
        X_validation=validation[feature_columns],
        y_validation=y_validation,
        groups_validation=groups_validation,
        cv=cv,
        bootstrap_reps=20,
        min_sensitivity=0.90,
    )

    threshold = result["threshold_selection"]["f2_threshold"]
    assert 0 <= threshold <= 1
    assert result["validation"]["metrics"]["threshold"] == pytest.approx(threshold)
    assert result["model"] == "logistic_regression"
    assert "_pipeline" in result
    assert "_pipeline" not in public_candidate_result(result)

    scores = result["_pipeline"].predict_proba(validation[feature_columns])
    assert scores.shape == (len(validation), 2)


def test_tuning_supports_a_sampling_strategy() -> None:
    from sklearn.model_selection import StratifiedGroupKFold

    train, y_train, groups_train, _, _, _ = _toy_split()
    cv = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=config.SEED)
    spec = _toy_spec()
    feature_columns = list(config.ALL_FEATURES)

    search = tune_candidate(
        spec,
        "random_over",
        X=train[feature_columns],
        y=y_train,
        groups=groups_train,
        cv=cv,
        n_iter=1,
        seed=config.SEED,
    )

    assert "sampler" in search.best_estimator_.named_steps


def test_select_final_candidate_prefers_higher_average_precision() -> None:
    def result(name: str, ap: float, f2: float) -> dict:
        return {
            "model": name,
            "validation": {"metrics": {"average_precision": ap, "f2": f2}},
        }

    selected = select_final_candidate(
        [
            result("logistic_regression", 0.30, 0.40),
            result("random_forest", 0.50, 0.35),
            result("hist_gradient_boosting", 0.498, 0.60),
        ]
    )

    # HistGradientBoosting is within the default AP tolerance and has the higher F2.
    assert selected["model"] == "hist_gradient_boosting"


def test_artifact_roundtrip_and_metadata(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    train, y_train, _, _, _, _ = _toy_split()
    feature_columns = list(config.ALL_FEATURES)
    pipeline = build_candidate_pipeline(_toy_spec(), "none")
    pipeline.fit(train[feature_columns], y_train)

    metadata = {
        "phase": 5,
        "target": {"definition": "melanoma"},
        "model": {"name": "logistic_regression"},
    }
    path = artifact_path_for("melanoma", "logistic_regression")
    artifact, sidecar = save_artifact(pipeline, metadata, path)

    loaded_pipeline, loaded_metadata = load_artifact(artifact)
    assert sidecar.exists()
    assert loaded_metadata["target"]["definition"] == "melanoma"
    assert loaded_metadata["artifact_sha256"]
    assert loaded_pipeline.predict_proba(train[feature_columns]).shape == (len(train), 2)


def test_frozen_pointer_resolves_to_the_artifact(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    pointer = frozen_pointer_path("melanoma")
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(
        json.dumps({"artifact": "models/phase5_melanoma_logistic_regression.joblib"}),
        encoding="utf-8",
    )

    resolved = resolve_artifact_path("melanoma")

    assert resolved == config.PROJECT_ROOT / "models/phase5_melanoma_logistic_regression.joblib"


def test_one_shot_guard_blocks_existing_final_test_report(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "METRICS_DIR", tmp_path)
    metrics_path, summary_path = final_test_paths("melanoma")

    assert metrics_path.name == "phase5_final_test_melanoma.json"
    assert summary_path.name == "phase5_final_test_melanoma.md"
    assert assert_test_not_yet_evaluated("melanoma") == metrics_path

    metrics_path.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        assert_test_not_yet_evaluated("melanoma")

    assert assert_test_not_yet_evaluated("melanoma", force=True) == metrics_path
