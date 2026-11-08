from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from melanoma_detection import config
from melanoma_detection import experiment
from melanoma_detection.data import sha256_file
from melanoma_detection.experiment import (
    assert_split_isolation,
    assert_test_not_yet_evaluated,
    evaluate_at_threshold,
    evaluate_baseline,
    load_artifact,
    positive_scores,
    resolve_artifact_path,
    save_artifact,
    select_final_candidate,
    split_counts,
    public_result,
    threshold_bundle,
)
from melanoma_detection.model import baseline_specs


def _frame(n_rows: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    target = [1 if index % 4 == 0 else 0 for index in range(n_rows)]
    data: dict[str, object] = {
        "image_id": [f"img_{index}" for index in range(n_rows)],
        "lesion_id": [f"les_{index}" for index in range(n_rows)],
        "target": target,
        "split": ["train"] * (n_rows // 2)
        + ["validation"] * (n_rows // 4)
        + ["test"] * (n_rows - n_rows // 2 - n_rows // 4),
    }
    for offset, column in enumerate(config.NUMERIC_FEATURES):
        data[column] = rng.normal(size=n_rows) + offset * np.asarray(target)
    data["sex"] = rng.choice(["male", "female"], size=n_rows)
    data["localization"] = rng.choice(["back", "trunk"], size=n_rows)
    return pd.DataFrame(data)


def _candidate(model: str, ap: float, f2: float) -> dict[str, object]:
    return {
        "model": model,
        "validation": {"metrics": {"average_precision": ap, "f2": f2}},
    }


def test_positive_scores_accepts_one_and_two_dimensional_input() -> None:
    assert np.array_equal(positive_scores(np.array([0.1, 0.9])), [0.1, 0.9])
    assert np.array_equal(
        positive_scores(np.array([[0.9, 0.1], [0.2, 0.8]])), [0.1, 0.8]
    )


def test_positive_scores_rejects_multiclass_output() -> None:
    with pytest.raises(ValueError, match="two columns"):
        positive_scores(np.zeros((3, 4)))


def test_threshold_bundle_selects_f2_and_sensitivity_thresholds() -> None:
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_score = np.array([0.05, 0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.95])

    thresholds = threshold_bundle(y_true, y_score, min_sensitivity=0.90)

    assert 0 <= thresholds["f2_threshold"] <= 1
    assert thresholds["sensitivity_threshold"] <= thresholds["f2_threshold"]
    assert thresholds["sensitivity_target"] == 0.90


def test_evaluate_at_threshold_returns_metrics_and_clustered_cis() -> None:
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9, 0.4, 0.6])
    groups = np.array(["a", "a", "b", "b", "c", "c"])

    result = evaluate_at_threshold(
        y_true, y_score, groups, threshold=0.35, bootstrap_reps=20
    )

    assert result["confusion_matrix"] == {"tn": 2, "fp": 1, "fn": 0, "tp": 3}
    assert result["metrics"]["sensitivity"] == pytest.approx(1.0)
    assert "sensitivity" in result["confidence_intervals"]


def test_split_counts_and_isolation() -> None:
    frame = _frame()
    counts = split_counts(frame)

    assert counts["train"]["rows"] == 40
    assert counts["test"]["rows"] == 20
    assert_split_isolation(frame)

    leaked = frame.copy()
    leaked.loc[leaked["split"] == "test", "lesion_id"] = frame.loc[
        frame["split"] == "train", "lesion_id"
    ].iloc[0]
    with pytest.raises(RuntimeError, match="lesions overlap"):
        assert_split_isolation(leaked)


def test_save_and_load_artifact_round_trip(tmp_path) -> None:
    path = tmp_path / "model.joblib"
    artifact, sidecar = save_artifact({"kind": "toy"}, {"target": "melanoma"}, path)

    loaded, metadata = load_artifact(artifact)

    assert loaded == {"kind": "toy"}
    assert metadata["target"] == "melanoma"
    assert metadata["artifact_sha256"] == sha256_file(artifact)
    assert json.loads(sidecar.read_text())["artifact"].endswith("model.joblib")


def test_resolve_artifact_path_uses_explicit_path_and_pointer(
    tmp_path, monkeypatch
) -> None:
    explicit = tmp_path / "explicit.joblib"
    assert resolve_artifact_path(explicit) == explicit

    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    (tmp_path / "model.joblib").write_bytes(b"x")
    pointer = config.frozen_pointer_path()
    pointer.write_text(json.dumps({"artifact": str(tmp_path / "model.joblib")}))

    assert resolve_artifact_path(None) == tmp_path / "model.joblib"


def test_final_test_guard_refuses_a_second_run(tmp_path, monkeypatch) -> None:
    metrics = tmp_path / "final_test.json"
    monkeypatch.setattr(
        config, "final_test_paths", lambda: (metrics, tmp_path / "final_test.md")
    )

    assert assert_test_not_yet_evaluated() == metrics

    metrics.write_text("{}")
    with pytest.raises(FileExistsError, match="already exists"):
        assert_test_not_yet_evaluated()
    assert assert_test_not_yet_evaluated(force=True) == metrics


def test_select_final_candidate_prefers_average_precision_then_f2() -> None:
    results = [
        _candidate("logistic_regression", ap=0.30, f2=0.60),
        _candidate("random_forest", ap=0.35, f2=0.55),
        _candidate("extra_trees", ap=0.346, f2=0.58),
    ]

    selected = select_final_candidate(results)

    # ExtraTrees is within 0.005 AP of the best model and has the better F2.
    assert selected["model"] == "extra_trees"


def test_evaluate_baseline_handles_constant_and_fitted_models() -> None:
    frame = _frame()
    train = frame.loc[frame["split"] == "train"].reset_index(drop=True)
    validation = frame.loc[frame["split"] == "validation"].reset_index(drop=True)
    cv = 3

    for name in ("majority", "age_only"):
        spec = next(spec for spec in baseline_specs() if spec["name"] == name)
        result = public_result(
            evaluate_baseline(
                spec,
                train=train,
                validation=validation,
                cv=cv,
                bootstrap_reps=10,
                min_sensitivity=0.90,
            )
        )
        assert "validation" in result
        assert 0 <= result["validation"]["metrics"]["average_precision"] <= 1
