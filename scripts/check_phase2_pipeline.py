#!/usr/bin/env python3
"""Phase 2 smoke check: leak-free pipeline and grouped CV on the training split."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_val_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from melanoma_detection import config  # noqa: E402
from melanoma_detection.data import load_modeling_frame  # noqa: E402
from melanoma_detection.pipeline import build_pipeline, selected_feature_count  # noqa: E402
from melanoma_detection.splits import make_grouped_cv  # noqa: E402


def _parse_k(value: str) -> int | str | None:
    normalised = value.strip().lower()
    if normalised in {"none", "0", "off"}:
        return None
    if normalised == "all":
        return "all"
    parsed = int(normalised)
    if parsed < 1:
        raise argparse.ArgumentTypeError("k must be a positive integer, 'all', or 'none'")
    return parsed


def _json_default(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=sorted(config.TARGET_SPECS),
        default=config.TARGET_DEFINITION,
    )
    parser.add_argument("--n-splits", type=int, default=config.N_SPLITS)
    parser.add_argument("--k", type=_parse_k, default=config.DEFAULT_SELECT_K)
    parser.add_argument(
        "--scale-numeric",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Standardise numeric features (use --no-scale-numeric for trees).",
    )
    args = parser.parse_args()

    frame = load_modeling_frame(args.target)
    train = frame.loc[frame["split"] == "train"].reset_index(drop=True)
    validation = frame.loc[frame["split"] == "validation"].reset_index(drop=True)

    feature_columns = list(config.ALL_FEATURES)
    X_train = train[feature_columns]
    y_train = train["target"].to_numpy()
    groups = train[config.GROUP_COLUMN].to_numpy()

    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=config.SEED,
    )
    pipeline = build_pipeline(
        classifier,
        scale_numeric=args.scale_numeric,
        k=args.k,
    )

    # Average precision is a binary-ranking metric. For the seven-class task,
    # balanced accuracy is the more meaningful smoke metric.
    if args.target == "multiclass":
        scoring = "balanced_accuracy"
        cv_name = "balanced accuracy"
    else:
        scoring = "average_precision"
        cv_name = "average precision"

    cv = make_grouped_cv(n_splits=args.n_splits, seed=config.SEED)
    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        groups=groups,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
    )

    pipeline.fit(X_train, y_train)
    validation_predictions = pipeline.predict(validation[feature_columns])

    preprocessed_features = len(pipeline.named_steps["preprocessor"].get_feature_names_out())
    selected_features = selected_feature_count(pipeline)

    if args.target == "multiclass":
        validation_smoke = {
            "balanced_accuracy": float(
                balanced_accuracy_score(validation["target"], validation_predictions)
            ),
            "note": "Smoke check only; validation is not used for model selection in Phase 2.",
        }
    else:
        validation_smoke = {
            "predicted_positive_rate": float(np.mean(validation_predictions == 1)),
            "note": "Smoke check only; validation is not used for model selection in Phase 2.",
        }

    result = {
        "phase": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": args.target,
        "model": "LogisticRegression(class_weight='balanced')",
        "pipeline": {
            "scale_numeric": args.scale_numeric,
            "k": args.k,
            "preprocessed_feature_count": preprocessed_features,
            "selected_feature_count": selected_features,
        },
        "data": {
            "train_rows": int(len(train)),
            "train_lesions": int(train[config.GROUP_COLUMN].nunique()),
            "validation_rows": int(len(validation)),
            "validation_lesions": int(validation[config.GROUP_COLUMN].nunique()),
        },
        "grouped_cv": {
            "n_splits": args.n_splits,
            "seed": config.SEED,
            "scoring": scoring,
            "scores": [float(score) for score in cv_scores],
            "mean": float(np.mean(cv_scores)),
            "std": float(np.std(cv_scores)),
        },
        "validation_smoke": validation_smoke,
    }

    output_path = config.METRICS_DIR / f"phase2_smoke_{args.target}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )

    print("Phase 2 smoke check")
    print(f"Target: {args.target}")
    print(f"Train rows/lesions: {len(train)} / {train[config.GROUP_COLUMN].nunique()}")
    print(f"Validation rows/lesions: {len(validation)} / {validation[config.GROUP_COLUMN].nunique()}")
    print(
        "Pipeline: preprocessor -> "
        + (f"SelectKBest(k={args.k}) -> " if args.k is not None else "")
        + "LogisticRegression"
    )
    print(
        f"Grouped CV {cv_name}: {np.mean(cv_scores):.4f} "
        f"(+/- {np.std(cv_scores):.4f})"
    )
    print(f"Preprocessed features: {preprocessed_features}; selected: {selected_features}")
    if args.target == "multiclass":
        print(f"Validation balanced accuracy: {validation_smoke['balanced_accuracy']:.3f}")
    else:
        print(
            f"Validation predicted-positive rate: "
            f"{validation_smoke['predicted_positive_rate']:.3f}"
        )
    print(f"Wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
