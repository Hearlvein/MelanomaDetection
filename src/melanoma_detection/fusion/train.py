"""Fit, freeze, and persist the tabular + embedding fusion model."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import config
from ..data import sha256_file
from ..evaluate import binary_metrics
from ..splits import make_grouped_cv
from ..train import load_artifact, save_artifact
from ..training import (
    evaluate_at_threshold,
    oof_positive_scores,
    positive_scores,
    threshold_bundle,
)
from .pipeline import (
    build_fusion_pipeline,
    fusion_feature_columns,
    fusion_pointer_path,
)


def fusion_scores(
    pipeline: Any,
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> np.ndarray:
    return positive_scores(pipeline.predict_proba(frame[feature_columns]))


def fit_fusion(
    frame: pd.DataFrame,
    *,
    numeric_features: list[str],
    categorical_features: list[str],
    embedding_features: list[str],
    n_components: int = config.FUSION_N_COMPONENTS,
    classifier: Any | None = None,
    n_splits: int = config.N_SPLITS,
    seed: int = config.SEED,
    bootstrap_reps: int = 2_000,
    min_sensitivity: float = 0.90,
) -> tuple[Any, dict[str, Any], list[str]]:
    """Fit the fusion pipeline and freeze its validation threshold.

    The threshold comes from grouped out-of-fold scores inside the training
    split. The validation split is used once for the reported validation
    metrics; the test split is never touched here.
    """
    feature_columns = fusion_feature_columns(
        numeric_features, categorical_features, embedding_features
    )
    train = frame.loc[frame["split"] == "train"].reset_index(drop=True)
    validation = frame.loc[frame["split"] == "validation"].reset_index(drop=True)
    if train.empty or validation.empty:
        raise ValueError("train and validation splits must both be non-empty")

    pipeline = build_fusion_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        embedding_features=embedding_features,
        n_components=n_components,
        classifier=classifier,
        seed=seed,
    )

    cv = make_grouped_cv(n_splits=n_splits, seed=seed)
    oof_scores = oof_positive_scores(
        pipeline,
        train[feature_columns],
        train["target"].to_numpy(),
        train["lesion_id"].to_numpy(),
        cv,
    )
    thresholds = threshold_bundle(
        train["target"].to_numpy(), oof_scores, min_sensitivity=min_sensitivity
    )
    oof_metrics = binary_metrics(
        train["target"].to_numpy(),
        oof_scores,
        threshold=thresholds["f2_threshold"],
    )

    pipeline.fit(train[feature_columns], train["target"].to_numpy())
    validation_scores = fusion_scores(pipeline, validation, feature_columns)
    validation_evaluation = evaluate_at_threshold(
        validation["target"].to_numpy(),
        validation_scores,
        validation["lesion_id"].to_numpy(),
        threshold=thresholds["f2_threshold"],
        bootstrap_reps=bootstrap_reps,
        seed=seed,
    )

    high_sensitivity = None
    if thresholds["sensitivity_threshold"] is not None:
        high_sensitivity = binary_metrics(
            validation["target"].to_numpy(),
            validation_scores,
            threshold=thresholds["sensitivity_threshold"],
        )

    result = {
        "model": config.FUSION_MODEL_NAME,
        "n_components": int(n_components),
        "numeric_features": list(numeric_features),
        "categorical_features": list(categorical_features),
        "embedding_features": list(embedding_features),
        "threshold_selection": thresholds,
        "oof": {
            "average_precision": float(oof_metrics["average_precision"]),
            "roc_auc": float(oof_metrics["roc_auc"]),
            "f2": float(oof_metrics["f2"]),
        },
        "validation": validation_evaluation,
        "validation_at_sensitivity_threshold": high_sensitivity,
    }
    return pipeline, result, feature_columns


def save_fusion_artifact(
    pipeline: Any,
    metadata: dict[str, Any],
    path: str | Path,
) -> tuple[Path, Path]:
    return save_artifact(pipeline, metadata, path)


def load_fusion_artifact(path: str | Path) -> tuple[Any, dict[str, Any]]:
    return load_artifact(path)


def resolve_fusion_artifact(
    target: str = config.TARGET_DEFINITION,
    artifact: str | Path | None = None,
) -> Path:
    if artifact is not None:
        path = Path(artifact)
        return path if path.is_absolute() else config.PROJECT_ROOT / path

    pointer_path = fusion_pointer_path(target)
    if not pointer_path.exists():
        raise FileNotFoundError(
            f"frozen fusion pointer not found: {pointer_path}; "
            "run scripts/train_fusion.py first"
        )
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    path = Path(pointer["artifact"])
    return path if path.is_absolute() else config.PROJECT_ROOT / path


def write_fusion_pointer(
    target: str,
    path: str | Path,
    metadata: dict[str, Any],
) -> Path:
    pointer_path = fusion_pointer_path(target)
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "target": target,
        "model": config.FUSION_MODEL_NAME,
        "artifact": _relative(Path(path)),
        "artifact_sha256": sha256_file(path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_components": metadata.get("n_components"),
        "feature_version": metadata.get("feature_version"),
        "split_fingerprint": metadata.get("split_fingerprint"),
    }
    pointer_path.write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return pointer_path


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(config.PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)
