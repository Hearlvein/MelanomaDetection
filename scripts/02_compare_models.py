#!/usr/bin/env python3
"""Step 2: compare baselines and tuned models, then optionally freeze one.

Hyperparameters are tuned with grouped cross-validation inside ``split=train``.
The threshold comes from out-of-fold predictions on the training split.
Validation is used for model comparison; the test split is never loaded.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from melanoma_detection import config  # noqa: E402
from melanoma_detection.data import load_modeling_frame, make_grouped_cv, split_fingerprint  # noqa: E402
from melanoma_detection.experiment import (  # noqa: E402
    assert_split_isolation,
    evaluate_baseline,
    evaluate_candidate,
    public_result,
    relative_path,
    sanitise_json,
    save_artifact,
    select_final_candidate,
    split_counts,
    tune_candidate,
)
from melanoma_detection.model import (  # noqa: E402
    DEFAULT_STRATEGIES,
    MODEL_ORDER,
    baseline_specs,
    candidate_specs,
    strategies_for_model,
)


def _target_info() -> dict[str, Any]:
    spec = config.TARGET_SPECS[config.TARGET_DEFINITION]
    return {
        "definition": config.TARGET_DEFINITION,
        "description": spec["description"],
        "positive_classes": list(spec["positive"]),
        "negative_classes": list(spec["negative"]),
        "labels": {1: spec["positive_label"], 0: spec["negative_label"]},
    }


def _write_pointer(
    *,
    selected: dict[str, Any],
    artifact: Path,
    sidecar: Path,
    fingerprint: str,
    created_at: str,
) -> Path:
    pointer = config.frozen_pointer_path()
    pointer.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "target": config.TARGET_DEFINITION,
        "model": selected["model"],
        "artifact": relative_path(artifact),
        "sidecar": relative_path(sidecar),
        "threshold": selected["threshold_selection"],
        "validation": selected["validation"]["metrics"],
        "split_fingerprint": fingerprint,
        "created_at": created_at,
    }
    pointer.write_text(
        json.dumps(sanitise_json(payload), indent=2) + "\n", encoding="utf-8"
    )
    return pointer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-splits", type=int, default=config.N_SPLITS)
    parser.add_argument("--n-iter", type=int, default=30)
    parser.add_argument("--bootstrap-reps", type=int, default=2_000)
    parser.add_argument("--sensitivity", type=float, default=0.90)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_ORDER,
        default=list(MODEL_ORDER),
        help="Candidate model families to tune.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=DEFAULT_STRATEGIES,
        default=list(DEFAULT_STRATEGIES),
        help="Imbalance strategies to compare for each model.",
    )
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="Save the selected candidate as the frozen artifact.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow --freeze to replace an existing frozen artifact.",
    )
    args = parser.parse_args()

    frame = load_modeling_frame()
    assert_split_isolation(frame)

    train = frame.loc[frame["split"] == "train"].reset_index(drop=True)
    validation = frame.loc[frame["split"] == "validation"].reset_index(drop=True)
    if train.empty or validation.empty:
        raise RuntimeError("training and validation splits must both contain rows")

    cv = make_grouped_cv(n_splits=args.n_splits, seed=config.SEED)
    feature_columns = list(config.ALL_FEATURES)
    X_train = train[feature_columns]
    y_train = train["target"].to_numpy()
    groups_train = train[config.GROUP_COLUMN].to_numpy()
    X_validation = validation[feature_columns]
    y_validation = validation["target"].to_numpy()
    groups_validation = validation[config.GROUP_COLUMN].to_numpy()

    print("Evaluating baselines and feature subsets...")
    baselines = []
    for spec in baseline_specs():
        result = evaluate_baseline(
            spec,
            train=train,
            validation=validation,
            cv=cv,
            bootstrap_reps=args.bootstrap_reps,
            min_sensitivity=args.sensitivity,
        )
        baselines.append(public_result(result))
        print(
            f"  {spec['name']}: validation AP="
            f"{result['validation']['metrics']['average_precision']:.4f}"
        )

    candidates: list[dict[str, Any]] = []
    pipelines: dict[tuple[str, str], Any] = {}
    for spec in candidate_specs():
        if spec["name"] not in args.models:
            continue
        for strategy in args.strategies:
            if strategy not in strategies_for_model(spec["name"]):
                continue
            print(f"Tuning {spec['label']} / {strategy}...")
            try:
                search = tune_candidate(
                    spec,
                    strategy,
                    X_train,
                    y_train,
                    groups_train,
                    cv,
                    n_iter=args.n_iter,
                    seed=config.SEED,
                )
                result = evaluate_candidate(
                    search,
                    spec,
                    strategy,
                    X_train=X_train,
                    y_train=y_train,
                    groups_train=groups_train,
                    X_validation=X_validation,
                    y_validation=y_validation,
                    groups_validation=groups_validation,
                    cv=cv,
                    bootstrap_reps=args.bootstrap_reps,
                    min_sensitivity=args.sensitivity,
                )
                pipelines[(spec["name"], strategy)] = result["_pipeline"]
                print(
                    f"  best CV AP={result['cv']['mean_test_score']:.4f}, "
                    f"validation AP="
                    f"{result['validation']['metrics']['average_precision']:.4f}"
                )
            except Exception as error:  # noqa: BLE001 - report per-candidate failures
                result = {
                    "model": spec["name"],
                    "model_label": spec["label"],
                    "strategy": strategy,
                    "error": f"{type(error).__name__}: {error}",
                }
                print(f"  failed: {result['error']}")
            candidates.append(public_result(result))

    successful = [result for result in candidates if "error" not in result]
    if not successful:
        raise RuntimeError("every candidate model failed")

    selected = select_final_candidate(successful)
    selected_pipeline = pipelines[(selected["model"], selected["strategy"])]
    fingerprint = split_fingerprint(frame, config.TARGET_DEFINITION, config.SEED)
    data = split_counts(frame)
    created_at = datetime.now(timezone.utc).isoformat()

    command = (
        "uv run --python 3.13 --with pandas --with openpyxl "
        "--with scikit-learn==1.9.1 --with imbalanced-learn "
        "python scripts/02_compare_models.py "
        f"--n-splits {args.n_splits} --n-iter {args.n_iter} "
        f"--bootstrap-reps {args.bootstrap_reps} --sensitivity {args.sensitivity}"
    )
    if args.freeze:
        command += " --freeze"

    artifact_info = None
    if args.freeze:
        artifact_path = config.artifact_path(selected["model"])
        pointer_path = config.frozen_pointer_path()
        if not args.force and (artifact_path.exists() or pointer_path.exists()):
            raise SystemExit(
                f"refusing to overwrite frozen artifact: {artifact_path}. "
                "Pass --force only if you intentionally want a new freeze."
            )

        metadata = {
            "stage": "frozen_model",
            "target": _target_info(),
            "model": {
                "name": selected["model"],
                "label": selected["model_label"],
                "scale_numeric": selected["scale_numeric"],
            },
            "imbalance_strategy": selected["strategy"],
            "best_params": selected["best_params"],
            "cv": selected["cv"],
            "oof": selected["oof"],
            "threshold": selected["threshold_selection"],
            "validation": selected["validation"],
            "validation_at_sensitivity_threshold": selected[
                "validation_at_sensitivity_threshold"
            ],
            "feature_columns": feature_columns,
            "preprocessing": {
                "numeric_imputer": "median",
                "categorical_imputer": "most_frequent",
                "categorical_encoder": "OneHotEncoder(handle_unknown='ignore')",
                "selector": "SelectKBest(f_classif)",
                "sampler": (
                    selected["strategy"]
                    if selected["strategy"] in {"random_over", "random_under", "smote"}
                    else None
                ),
            },
            "training": data["train"],
            "validation_data": data["validation"],
            "split": {
                "fingerprint": fingerprint,
                "seed": config.SEED,
                "n_splits": args.n_splits,
                "test_fold": config.TEST_FOLD,
                "val_fold": config.VAL_FOLD,
                "group_column": config.GROUP_COLUMN,
            },
            "software": {
                "python": sys.version.split()[0],
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "scikit-learn": sklearn.__version__,
            },
            "command": command,
            "notes": [
                "Threshold selected from grouped-CV OOF predictions on split == 'train'.",
                "Validation used for model comparison only.",
                "Test split not used during model selection.",
                "Not clinical evidence; research use only.",
            ],
        }
        artifact, sidecar = save_artifact(selected_pipeline, metadata, artifact_path)
        pointer = _write_pointer(
            selected=selected,
            artifact=artifact,
            sidecar=sidecar,
            fingerprint=fingerprint,
            created_at=created_at,
        )
        artifact_info = {
            "path": relative_path(artifact),
            "sidecar": relative_path(sidecar),
            "pointer": relative_path(pointer),
            "sha256": json.loads(sidecar.read_text(encoding="utf-8"))[
                "artifact_sha256"
            ],
        }
        print(f"Froze model: {artifact}")

    protocol = {
        "created_at": created_at,
        "seed": config.SEED,
        "n_splits": args.n_splits,
        "n_iter": args.n_iter,
        "bootstrap_reps": args.bootstrap_reps,
        "sensitivity_target": args.sensitivity,
        "threshold_source": "grouped CV OOF predictions on split == train",
        "selection_rule": "highest validation average precision; F2 tie-break within 0.005",
        "test_used": False,
    }
    payload = {
        "stage": "model_comparison",
        "target": _target_info(),
        "protocol": protocol,
        "data": {
            "train": data["train"],
            "validation": data["validation"],
            "test_used": False,
        },
        "baselines": baselines,
        "candidates": candidates,
        "selected": selected,
        "artifact": artifact_info,
        "software": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "command": command,
    }

    report_path = config.model_comparison_path()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(sanitise_json(payload), indent=2) + "\n", encoding="utf-8"
    )

    print()
    print(f"Wrote comparison: {report_path}")
    print(
        "Selected: "
        f"{selected['model_label']} / {selected['strategy']} "
        f"(validation AP="
        f"{selected['validation']['metrics']['average_precision']:.3f}, "
        f"F2={selected['validation']['metrics']['f2']:.3f}, "
        f"threshold={selected['threshold_selection']['f2_threshold']:.3f})"
    )
    print("Test split was not used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
