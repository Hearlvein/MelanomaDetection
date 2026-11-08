"""Tuning, model selection, threshold freezing, and artifact helpers.

The test split is never used here. Hyperparameters are tuned with grouped CV
inside ``split == "train"``; the threshold comes from out-of-fold predictions on
that same split; validation is used only to compare already-tuned candidates.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict

from . import config
from .data import sha256_file
from .evaluate import (
    binary_metrics,
    cluster_bootstrap_metric_cis,
    select_f2_threshold,
    select_sensitivity_threshold,
)
from .model import MODEL_ORDER, build_baseline_pipeline, build_candidate_pipeline


# ---------------------------------------------------------------------------
# Scores and metrics
# ---------------------------------------------------------------------------

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
    """Generate grouped out-of-fold positive-class scores on the training split."""
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
    """Select the primary F2 threshold and the secondary sensitivity threshold."""
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
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        shared = lesion_sets[left] & lesion_sets[right]
        if shared:
            raise RuntimeError(
                f"{len(shared)} lesions overlap between {left} and {right}; "
                "rerun 01_dataset.py before evaluating."
            )


# ---------------------------------------------------------------------------
# Tuning and validation
# ---------------------------------------------------------------------------

def tune_candidate(
    spec: dict[str, Any],
    strategy: str,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    cv,
    *,
    n_iter: int = 30,
    seed: int = config.SEED,
) -> RandomizedSearchCV:
    """Tune one candidate with grouped CV on the training split only."""
    pipeline = build_candidate_pipeline(spec, strategy)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=spec["param_distributions"],
        n_iter=n_iter,
        scoring="average_precision",
        cv=cv,
        refit=True,
        random_state=seed,
        n_jobs=1,
        verbose=0,
        error_score="raise",
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="'penalty' was deprecated.*", category=FutureWarning
        )
        warnings.filterwarnings(
            "ignore", message="Inconsistent values: penalty=.*", category=UserWarning
        )
        search.fit(X, y, groups=groups)
    return search


def evaluate_candidate(
    search: RandomizedSearchCV,
    spec: dict[str, Any],
    strategy: str,
    *,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_validation: pd.DataFrame,
    y_validation: np.ndarray,
    groups_validation: np.ndarray,
    cv,
    bootstrap_reps: int,
    min_sensitivity: float,
) -> dict[str, Any]:
    """Freeze a threshold from OOF scores, refit, and evaluate validation once."""
    best_estimator = search.best_estimator_
    oof_scores = oof_positive_scores(best_estimator, X_train, y_train, groups_train, cv)
    thresholds = threshold_bundle(y_train, oof_scores, min_sensitivity=min_sensitivity)
    oof_metrics = binary_metrics(
        y_train, oof_scores, threshold=thresholds["f2_threshold"]
    )

    best_estimator.fit(X_train, y_train)
    validation_scores = positive_scores(best_estimator.predict_proba(X_validation))
    evaluation = evaluate_at_threshold(
        y_validation,
        validation_scores,
        groups_validation,
        threshold=thresholds["f2_threshold"],
        bootstrap_reps=bootstrap_reps,
        seed=config.SEED,
    )

    high_sensitivity = None
    if thresholds["sensitivity_threshold"] is not None:
        high_sensitivity = binary_metrics(
            y_validation,
            validation_scores,
            threshold=thresholds["sensitivity_threshold"],
        )

    best_index = int(search.best_index_)
    return {
        "model": spec["name"],
        "model_label": spec["label"],
        "strategy": strategy,
        "scale_numeric": spec["scale_numeric"],
        "best_params": _jsonify_params(search.best_params_),
        "cv": {
            "scoring": "average_precision",
            "best_mean_score": float(search.best_score_),
            "mean_test_score": float(search.cv_results_["mean_test_score"][best_index]),
            "std_test_score": float(search.cv_results_["std_test_score"][best_index]),
            "rank_test_score": int(search.cv_results_["rank_test_score"][best_index]),
            "n_candidates": int(len(search.cv_results_["params"])),
        },
        "threshold_selection": thresholds,
        "oof": {
            "average_precision": float(oof_metrics["average_precision"]),
            "roc_auc": float(oof_metrics["roc_auc"]),
            "f2": float(oof_metrics["f2"]),
        },
        "validation": evaluation,
        "validation_at_sensitivity_threshold": high_sensitivity,
        "_pipeline": best_estimator,
    }


def evaluate_baseline(
    spec: dict[str, Any],
    *,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    cv,
    bootstrap_reps: int,
    min_sensitivity: float,
) -> dict[str, Any]:
    """Evaluate one constant or single-feature-group baseline on validation."""
    feature_columns = list(config.ALL_FEATURES)
    X_train = train[feature_columns]
    y_train = train["target"].to_numpy()
    groups_train = train[config.GROUP_COLUMN].to_numpy()
    X_validation = validation[feature_columns]
    y_validation = validation["target"].to_numpy()
    groups_validation = validation[config.GROUP_COLUMN].to_numpy()

    pipeline = None
    if spec["kind"] == "constant":
        oof_scores = np.full(len(train), float(spec["score"]), dtype=float)
        validation_scores = np.full(len(validation), float(spec["score"]), dtype=float)
        thresholds = {"f2_threshold": 0.5, "sensitivity_target": None, "sensitivity_threshold": None}
    elif spec["kind"] == "random":
        rng = np.random.default_rng(config.SEED)
        oof_scores = rng.random(len(train))
        validation_scores = rng.random(len(validation))
        thresholds = {"f2_threshold": 0.5, "sensitivity_target": None, "sensitivity_threshold": None}
    else:
        pipeline = build_baseline_pipeline(spec)
        oof_scores = oof_positive_scores(pipeline, X_train, y_train, groups_train, cv)
        thresholds = threshold_bundle(
            y_train, oof_scores, min_sensitivity=min_sensitivity
        )
        pipeline.fit(X_train, y_train)
        validation_scores = positive_scores(pipeline.predict_proba(X_validation))

    evaluation = evaluate_at_threshold(
        y_validation,
        validation_scores,
        groups_validation,
        threshold=float(thresholds["f2_threshold"]),
        bootstrap_reps=bootstrap_reps,
        seed=config.SEED,
    )

    high_sensitivity = None
    if thresholds["sensitivity_threshold"] is not None:
        high_sensitivity = binary_metrics(
            y_validation,
            validation_scores,
            threshold=float(thresholds["sensitivity_threshold"]),
        )

    return {
        "name": spec["name"],
        "description": spec["description"],
        "kind": spec["kind"],
        "features": {
            "numeric": list(spec["numeric_features"]),
            "categorical": list(spec["categorical_features"]),
            "numeric_count": len(spec["numeric_features"]),
            "categorical_count": len(spec["categorical_features"]),
        },
        "threshold_selection": {
            "source": "grouped CV out-of-fold predictions on split == train",
            "objective": "maximise F2",
            **thresholds,
        },
        "validation": evaluation,
        "validation_at_sensitivity_threshold": high_sensitivity,
        "_pipeline": pipeline,
    }


def select_final_candidate(
    results: list[dict[str, Any]],
    *,
    average_precision_tolerance: float = 0.005,
) -> dict[str, Any]:
    """Pick the final candidate from validation performance.

    Candidates within ``average_precision_tolerance`` of the best validation
    average precision are compared on F2. Ties prefer the earlier model family
    in :data:`model.MODEL_ORDER`.
    """
    successful = [result for result in results if "error" not in result]
    if not successful:
        raise RuntimeError("no candidate model was evaluated successfully")

    best_ap = max(
        result["validation"]["metrics"]["average_precision"] for result in successful
    )
    close = [
        result
        for result in successful
        if best_ap - result["validation"]["metrics"]["average_precision"]
        <= average_precision_tolerance
    ]

    def key(result: dict[str, Any]) -> tuple[float, int]:
        order = MODEL_ORDER.index(result["model"])
        return (result["validation"]["metrics"]["f2"], -order)

    return max(close, key=key)


def public_result(result: dict[str, Any]) -> dict[str, Any]:
    """Drop non-serialisable fields from a candidate result."""
    return {key: value for key, value in result.items() if not key.startswith("_")}


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------

def save_artifact(
    pipeline: Any,
    metadata: dict[str, Any],
    path: str | Path,
) -> tuple[Path, Path]:
    """Persist a fitted pipeline with a JSON sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)

    payload = dict(metadata)
    payload["created_at"] = datetime.now(timezone.utc).isoformat()
    payload["artifact"] = relative_path(path)
    payload["artifact_sha256"] = sha256_file(path)

    sidecar_path = path.with_suffix(".json")
    sidecar_path.write_text(
        json.dumps(sanitise_json(payload), indent=2) + "\n", encoding="utf-8"
    )
    return path, sidecar_path


def load_artifact(path: str | Path) -> tuple[Any, dict[str, Any]]:
    """Load a frozen pipeline and its sidecar."""
    path = Path(path)
    sidecar_path = path.with_suffix(".json")
    if not path.exists():
        raise FileNotFoundError(f"model artifact not found: {path}")
    if not sidecar_path.exists():
        raise FileNotFoundError(f"model sidecar not found: {sidecar_path}")

    pipeline = joblib.load(path)
    metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
    return pipeline, metadata


def resolve_artifact_path(artifact: str | Path | None = None) -> Path:
    """Resolve an explicit artifact or the frozen pointer."""
    if artifact is not None:
        path = Path(artifact)
        return path if path.is_absolute() else config.PROJECT_ROOT / path

    pointer_path = config.frozen_pointer_path()
    if not pointer_path.exists():
        raise FileNotFoundError(
            f"frozen artifact pointer not found: {pointer_path}; "
            "run 02_compare_models.py --freeze first"
        )
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    path = Path(pointer["artifact"])
    return path if path.is_absolute() else config.PROJECT_ROOT / path


def assert_test_not_yet_evaluated(*, force: bool = False) -> Path:
    """Refuse to overwrite an existing final-test report unless forced."""
    metrics_path, _ = config.final_test_paths()
    if metrics_path.exists() and not force:
        raise FileExistsError(
            f"final test report already exists: {metrics_path}. "
            "The test split must be evaluated once. Pass --force only if you "
            "intentionally need to overwrite the frozen result."
        )
    return metrics_path


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(config.PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def sanitise_json(value: Any) -> Any:
    """Recursively replace non-finite floats so the output is strict JSON."""
    if isinstance(value, dict):
        return {key: sanitise_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitise_json(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _jsonify_params(params: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in params.items():
        result[key] = value.item() if isinstance(value, np.generic) else value
    return result
