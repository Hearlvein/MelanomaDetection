"""Phase 5 helpers: tuned candidates, threshold freezing, and artifacts.

The test split is never used here. Model selection is performed on the
validation split after hyperparameters are tuned with grouped CV inside the
training split.
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
from scipy.stats import loguniform
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

from . import config
from .data import sha256_file
from .evaluate import binary_metrics
from .imbalance import build_imbalance_pipeline, class_weight_for
from .training import (
    evaluate_at_threshold,
    oof_positive_scores,
    positive_scores,
    threshold_bundle,
)

CANDIDATE_ORDER = (
    "logistic_regression",
    "random_forest",
    "hist_gradient_boosting",
    "extra_trees",
)
SELECT_K_VALUES = (10, 15, 20, 25, 30)


def _logistic_regression(class_weight: str | None) -> LogisticRegression:
    return LogisticRegression(
        max_iter=5_000,
        solver="liblinear",
        class_weight=class_weight,
        random_state=config.SEED,
    )


def _random_forest(class_weight: str | None) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight=class_weight,
        random_state=config.SEED,
        n_jobs=-1,
    )


def _hist_gradient_boosting(class_weight: str | None) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.1,
        early_stopping=False,
        class_weight=class_weight,
        random_state=config.SEED,
    )


def _extra_trees(class_weight: str | None) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        class_weight=class_weight,
        random_state=config.SEED,
        n_jobs=-1,
    )


def candidate_specs(
    strategy_overrides: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Return the small Phase 5 model set with search spaces.

    ``strategy_overrides`` maps a model name to the imbalance strategy chosen in
    Phase 4. Models without an override use ``default_strategy``.
    """
    overrides = dict(strategy_overrides or {})
    selector_space = list(SELECT_K_VALUES)

    specs: list[dict[str, Any]] = [
        {
            "name": "logistic_regression",
            "label": "Logistic Regression",
            "estimator_factory": _logistic_regression,
            "scale_numeric": True,
            "default_strategy": "balanced",
            "param_distributions": {
                "classifier__C": loguniform(1e-3, 1e2),
                "classifier__penalty": ["l1", "l2"],
                "selector__k": selector_space,
            },
        },
        {
            "name": "random_forest",
            "label": "Random Forest",
            "estimator_factory": _random_forest,
            "scale_numeric": False,
            "default_strategy": "balanced",
            "param_distributions": {
                "classifier__n_estimators": [200, 400],
                "classifier__max_depth": [None, 10, 20, 30],
                "classifier__min_samples_leaf": [1, 2, 4, 8],
                "classifier__max_features": ["sqrt", "log2", None],
                "selector__k": selector_space,
            },
        },
        {
            "name": "hist_gradient_boosting",
            "label": "HistGradientBoosting",
            "estimator_factory": _hist_gradient_boosting,
            "scale_numeric": False,
            "default_strategy": "balanced",
            "param_distributions": {
                "classifier__learning_rate": [0.03, 0.05, 0.1],
                "classifier__max_iter": [100, 200, 300],
                "classifier__max_leaf_nodes": [15, 31, 63],
                "classifier__min_samples_leaf": [10, 20, 40],
                "classifier__l2_regularization": [0.0, 0.1, 1.0],
                "selector__k": selector_space,
            },
        },
        {
            "name": "extra_trees",
            "label": "ExtraTrees",
            "estimator_factory": _extra_trees,
            "scale_numeric": False,
            "default_strategy": "balanced",
            "param_distributions": {
                "classifier__n_estimators": [300, 500],
                "classifier__max_depth": [None, 10, 20, 30],
                "classifier__min_samples_leaf": [1, 2, 4],
                "classifier__max_features": ["sqrt", "log2", None],
                "selector__k": selector_space,
            },
        },
    ]

    for spec in specs:
        spec["strategy"] = overrides.get(spec["name"], spec["default_strategy"])
    return specs


def build_candidate_pipeline(spec: dict[str, Any], strategy: str):
    """Build the tuned pipeline for one candidate and imbalance strategy."""
    class_weight = class_weight_for(strategy, spec["name"])
    classifier = spec["estimator_factory"](class_weight)
    return build_imbalance_pipeline(
        classifier,
        strategy=strategy,
        scale_numeric=spec["scale_numeric"],
        k=config.DEFAULT_SELECT_K,
        seed=config.SEED,
    )


def tune_candidate(
    spec: dict[str, Any],
    strategy: str,
    X,
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
    # scikit-learn 1.9 deprecates LogisticRegression's ``penalty`` in favour of
    # ``l1_ratio``. Keep the search space compatible with the project's wider
    # supported range while suppressing only those specific warnings.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="'penalty' was deprecated.*",
            category=FutureWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Inconsistent values: penalty=.*",
            category=UserWarning,
        )
        search.fit(X, y, groups=groups)
    return search


def evaluate_tuned_candidate(
    search: RandomizedSearchCV,
    spec: dict[str, Any],
    strategy: str,
    *,
    X_train,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_validation,
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

    # Refit explicitly on the full training split. The threshold was selected
    # from OOF scores and must not influence this fit.
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


def select_final_candidate(
    results: list[dict[str, Any]],
    *,
    average_precision_tolerance: float = 0.005,
) -> dict[str, Any]:
    """Pick the final candidate from validation performance.

    Candidates within ``average_precision_tolerance`` of the best validation
    average precision are compared on F2. Ties prefer the simpler / earlier
    model family in ``CANDIDATE_ORDER``.
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
        order = CANDIDATE_ORDER.index(result["model"])
        return (result["validation"]["metrics"]["f2"], -order)

    return max(close, key=key)


def public_candidate_result(result: dict[str, Any]) -> dict[str, Any]:
    """Drop non-serialisable fields from a candidate result."""
    return {key: value for key, value in result.items() if not key.startswith("_")}


def save_artifact(
    pipeline,
    metadata: dict[str, Any],
    path: str | Path,
) -> tuple[Path, Path]:
    """Persist a fitted pipeline with a JSON sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, path)

    payload = dict(metadata)
    payload["created_at"] = datetime.now(timezone.utc).isoformat()
    payload["artifact"] = _relative(path)
    payload["artifact_sha256"] = sha256_file(path)

    sidecar_path = path.with_suffix(".json")
    sidecar_path.write_text(
        json.dumps(_sanitise_json(payload), indent=2) + "\n",
        encoding="utf-8",
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


def artifact_path_for(target: str, model: str) -> Path:
    return config.MODELS_DIR / f"phase5_{target}_{model}.joblib"


def frozen_pointer_path(target: str) -> Path:
    return config.MODELS_DIR / f"phase5_{target}_frozen.json"


def resolve_artifact_path(target: str, artifact: str | Path | None = None) -> Path:
    """Resolve an explicit artifact or the target's frozen pointer."""
    if artifact is not None:
        path = Path(artifact)
        return path if path.is_absolute() else config.PROJECT_ROOT / path

    pointer_path = frozen_pointer_path(target)
    if not pointer_path.exists():
        raise FileNotFoundError(
            f"frozen artifact pointer not found: {pointer_path}; "
            "run scripts/run_phase5_model_selection.py first"
        )

    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    path = Path(pointer["artifact"])
    return path if path.is_absolute() else config.PROJECT_ROOT / path


def final_test_paths(target: str) -> tuple[Path, Path]:
    return (
        config.METRICS_DIR / f"phase5_final_test_{target}.json",
        config.METRICS_DIR / f"phase5_final_test_{target}.md",
    )


def assert_test_not_yet_evaluated(target: str, *, force: bool = False) -> Path:
    """Refuse to overwrite an existing final-test report unless forced."""
    metrics_path, _ = final_test_paths(target)
    if metrics_path.exists() and not force:
        raise FileExistsError(
            f"final test report already exists: {metrics_path}. "
            "The test split must be evaluated once. Pass --force only if you "
            "intentionally need to overwrite the frozen result."
        )
    return metrics_path


def _jsonify_params(params: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, np.generic):
            result[key] = value.item()
        else:
            result[key] = value
    return result


def _sanitise_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitise_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitise_json(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(config.PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)
