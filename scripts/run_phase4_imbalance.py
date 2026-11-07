#!/usr/bin/env python3
"""Phase 4: compare class-weight and resampling strategies under grouped CV."""

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
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from melanoma_detection import config  # noqa: E402
from melanoma_detection.data import load_modeling_frame  # noqa: E402
from melanoma_detection.evaluate import binary_metrics  # noqa: E402
from melanoma_detection.imbalance import (  # noqa: E402
    build_imbalance_pipeline,
    class_weight_for,
    strategies_for_model,
)
from melanoma_detection.splits import make_grouped_cv  # noqa: E402
from melanoma_detection.training import (  # noqa: E402
    assert_split_isolation,
    evaluate_at_threshold,
    oof_positive_scores,
    positive_scores,
    split_counts,
    threshold_bundle,
)

BINARY_TARGETS = ("melanoma", "malignant")
MODEL_ORDER = ("logistic_regression", "random_forest", "hist_gradient_boosting")


def _json_default(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return str(value)


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
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def _classifier(model_name: str, class_weight: str | None):
    if model_name == "logistic_regression":
        return LogisticRegression(
            max_iter=5_000,
            class_weight=class_weight,
            random_state=config.SEED,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight=class_weight,
            random_state=config.SEED,
            n_jobs=1,
        )
    if model_name == "hist_gradient_boosting":
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
    raise ValueError(f"unknown model: {model_name!r}")


def _model_label(model_name: str) -> str:
    return {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "hist_gradient_boosting": "HistGradientBoosting",
    }[model_name]


def _run_experiment(
    *,
    model_name: str,
    strategy: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    k: int,
    cv,
    bootstrap_reps: int,
    min_sensitivity: float,
) -> dict[str, Any]:
    feature_columns = list(config.ALL_FEATURES)
    X_train = train[feature_columns]
    y_train = train["target"].to_numpy()
    groups_train = train[config.GROUP_COLUMN].to_numpy()
    X_validation = validation[feature_columns]
    y_validation = validation["target"].to_numpy()

    class_weight = class_weight_for(strategy, model_name)
    classifier = _classifier(model_name, class_weight)
    pipeline = build_imbalance_pipeline(
        classifier,
        strategy=strategy,
        k=k,
        seed=config.SEED,
    )

    oof_scores = oof_positive_scores(pipeline, X_train, y_train, groups_train, cv)
    thresholds = threshold_bundle(y_train, oof_scores, min_sensitivity=min_sensitivity)
    oof_metrics = binary_metrics(y_train, oof_scores, threshold=thresholds["f2_threshold"])

    pipeline.fit(X_train, y_train)
    validation_scores = positive_scores(pipeline.predict_proba(X_validation))
    evaluation = evaluate_at_threshold(
        y_validation,
        validation_scores,
        validation[config.GROUP_COLUMN].to_numpy(),
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

    return {
        "model": model_name,
        "model_label": _model_label(model_name),
        "strategy": strategy,
        "class_weight": class_weight,
        "threshold_selection": thresholds,
        "oof_average_precision": float(oof_metrics["average_precision"]),
        "oof_f2_at_f2_threshold": float(oof_metrics["f2"]),
        "validation": evaluation,
        "validation_at_sensitivity_threshold": high_sensitivity,
    }


def _format_value(value: float | int | None) -> str:
    if value is None:
        return "—"
    number = float(value)
    if not np.isfinite(number):
        return "—"
    return f"{number:.3f}"


def _format_with_ci(result: dict[str, Any], metric: str) -> str:
    value = _format_value(result["validation"]["metrics"].get(metric))
    interval = result["validation"]["confidence_intervals"].get(metric)
    if interval is None or value == "—":
        return value
    return f"{value} [{interval['lower']:.3f}–{interval['upper']:.3f}]"


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _recommendations(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, dict[str, Any]] = {}
    for model_name in MODEL_ORDER:
        candidates = [result for result in results if result["model"] == model_name and "error" not in result]
        if not candidates:
            continue
        by_model[model_name] = max(
            candidates,
            key=lambda result: (
                result["validation"]["metrics"]["average_precision"],
                result["validation"]["metrics"]["f2"],
            ),
        )

    valid = [result for result in results if "error" not in result]
    global_best = max(
        valid,
        key=lambda result: (
            result["validation"]["metrics"]["average_precision"],
            result["validation"]["metrics"]["f2"],
        ),
    )
    return {
        "best_per_model": {
            model_name: {
                "strategy": result["strategy"],
                "validation_average_precision": result["validation"]["metrics"]["average_precision"],
                "validation_f2": result["validation"]["metrics"]["f2"],
            }
            for model_name, result in by_model.items()
        },
        "global_best": {
            "model": global_best["model"],
            "strategy": global_best["strategy"],
            "validation_average_precision": global_best["validation"]["metrics"]["average_precision"],
            "validation_f2": global_best["validation"]["metrics"]["f2"],
        },
    }


def _render_summary(
    *,
    target_info: dict[str, Any],
    protocol: dict[str, Any],
    data: dict[str, dict[str, Any]],
    results: list[dict[str, Any]],
    recommendations: dict[str, Any],
    command: str,
) -> str:
    successful = [result for result in results if "error" not in result]
    failed = [result for result in results if "error" in result]
    lines = [
        "# Phase 4 — Imbalance strategy comparison",
        "",
        f"- Generated at: `{protocol['created_at']}`",
        f"- Target: **{target_info['definition']}**",
        f"- Positive class: **{target_info['labels'][1]}** (`target = 1`)",
        f"- Negative class: **{target_info['labels'][0]}** (`target = 0`)",
        f"- Bootstrap: {protocol['bootstrap_reps']} lesion-clustered replicates, 95% percentile intervals",
        f"- Threshold source: `{protocol['threshold_source']}`",
        "",
        "## Protocol",
        "",
        "- Every model/strategy combination uses the Phase 2 preprocessor and `SelectKBest`.",
        "- Resampling happens inside the training fold only, after preprocessing and feature selection.",
        "- Thresholds are selected from grouped-CV OOF scores inside `split == \"train\"`.",
        "- The validation split is evaluated once per combination at the frozen threshold.",
        "- The test split is not used.",
        "",
        "## Data roles",
        "",
        _markdown_table(
            ["Split", "Rows", "Lesions", "Positive", "Negative", "Prevalence"],
            [
                [
                    split,
                    data[split]["rows"],
                    data[split]["lesions"],
                    data[split]["positive"],
                    data[split]["negative"],
                    _format_value(data[split]["prevalence"]),
                ]
                for split in ("train", "validation", "test")
            ],
        ),
        "",
        "## Results",
        "",
        _markdown_table(
            [
                "Model",
                "Strategy",
                "Threshold",
                "Sensitivity",
                "Specificity",
                "PPV",
                "NPV",
                "F1",
                "F2",
                "AP",
                "ROC-AUC",
                "Brier",
            ],
            [
                [
                    result["model_label"],
                    result["strategy"],
                    _format_value(result["threshold_selection"]["f2_threshold"]),
                    _format_with_ci(result, "sensitivity"),
                    _format_with_ci(result, "specificity"),
                    _format_with_ci(result, "ppv"),
                    _format_with_ci(result, "npv"),
                    _format_with_ci(result, "f1"),
                    _format_with_ci(result, "f2"),
                    _format_with_ci(result, "average_precision"),
                    _format_with_ci(result, "roc_auc"),
                    _format_with_ci(result, "brier_score"),
                ]
                for result in successful
            ],
        ),
        "",
        "## High-sensitivity operating points",
        "",
        _markdown_table(
            ["Model", "Strategy", "Threshold", "Sensitivity", "Specificity", "PPV", "NPV"],
            [
                [
                    result["model_label"],
                    result["strategy"],
                    _format_value(result["threshold_selection"]["sensitivity_threshold"]),
                    _format_value(result["validation_at_sensitivity_threshold"]["sensitivity"]),
                    _format_value(result["validation_at_sensitivity_threshold"]["specificity"]),
                    _format_value(result["validation_at_sensitivity_threshold"]["ppv"]),
                    _format_value(result["validation_at_sensitivity_threshold"]["npv"]),
                ]
                for result in successful
                if result["validation_at_sensitivity_threshold"] is not None
            ],
        ),
        "",
        *(
            [
                "## Failed combinations",
                "",
                _markdown_table(
                    ["Model", "Strategy", "Error"],
                    [
                        [result["model_label"], result["strategy"], result["error"]]
                        for result in failed
                    ],
                ),
                "",
            ]
            if failed
            else []
        ),
        "## Recommendation",
        "",
        "Best strategy per model by validation average precision, with F2 as a tie-breaker:",
        "",
        _markdown_table(
            ["Model", "Strategy", "Validation AP", "Validation F2"],
            [
                [
                    model_name,
                    details["strategy"],
                    _format_value(details["validation_average_precision"]),
                    _format_value(details["validation_f2"]),
                ]
                for model_name, details in recommendations["best_per_model"].items()
            ],
        ),
        "",
        (
            f"Global best: **{recommendations['global_best']['model']}** with "
            f"**{recommendations['global_best']['strategy']}** "
            f"(validation AP {_format_value(recommendations['global_best']['validation_average_precision'])}, "
            f"F2 {_format_value(recommendations['global_best']['validation_f2'])})."
        ),
        "",
        "This is a strategy comparison, not the final model selection. Phase 5 re-tunes a small"
        " model set using the chosen strategy and evaluates the test split only after freezing.",
        "",
        "## Reproduce",
        "",
        "```bash",
        command,
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=BINARY_TARGETS, default=config.TARGET_DEFINITION)
    parser.add_argument("--n-splits", type=int, default=config.N_SPLITS)
    parser.add_argument("--k", type=int, default=config.DEFAULT_SELECT_K)
    parser.add_argument("--bootstrap-reps", type=int, default=2_000)
    parser.add_argument("--sensitivity", type=float, default=0.90)
    args = parser.parse_args()

    frame = load_modeling_frame(args.target)
    assert_split_isolation(frame)
    train = frame.loc[frame["split"] == "train"].reset_index(drop=True)
    validation = frame.loc[frame["split"] == "validation"].reset_index(drop=True)
    if train.empty or validation.empty:
        raise RuntimeError("training and validation splits must both contain rows")

    spec = config.TARGET_SPECS[args.target]
    target_info = {
        "definition": args.target,
        "description": spec["description"],
        "positive_classes": list(spec["positive"]),
        "negative_classes": list(spec["negative"]),
        "labels": {1: spec["positive_label"], 0: spec["negative_label"]},
    }

    cv = make_grouped_cv(n_splits=args.n_splits, seed=config.SEED)
    results: list[dict[str, Any]] = []

    for model_name in MODEL_ORDER:
        for strategy in strategies_for_model(model_name):
            print(f"Evaluating {model_name} / {strategy}...")
            try:
                result = _run_experiment(
                    model_name=model_name,
                    strategy=strategy,
                    train=train,
                    validation=validation,
                    k=args.k,
                    cv=cv,
                    bootstrap_reps=args.bootstrap_reps,
                    min_sensitivity=args.sensitivity,
                )
            except Exception as error:  # noqa: BLE001 - report per-combination failures
                result = {
                    "model": model_name,
                    "model_label": _model_label(model_name),
                    "strategy": strategy,
                    "error": f"{type(error).__name__}: {error}",
                }
                print(f"  failed: {result['error']}")
            results.append(result)

    successful = [result for result in results if "error" not in result]
    if not successful:
        raise RuntimeError("all imbalance experiments failed")

    recommendations = _recommendations(results)
    data = split_counts(frame)
    created_at = datetime.now(timezone.utc).isoformat()
    protocol = {
        "created_at": created_at,
        "seed": config.SEED,
        "n_splits": args.n_splits,
        "k": args.k,
        "bootstrap_reps": args.bootstrap_reps,
        "sensitivity_target": args.sensitivity,
        "threshold_source": "grouped CV OOF predictions on split == train",
        "test_used": False,
    }

    metrics_path = config.METRICS_DIR / f"phase4_imbalance_{args.target}.json"
    summary_path = config.METRICS_DIR / f"phase4_imbalance_{args.target}.md"
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "phase": 4,
        "target": target_info,
        "protocol": protocol,
        "data": data,
        "results": results,
        "recommendations": recommendations,
        "software": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
        },
    }
    command = (
        "uv run --python 3.13 "
        "--with pandas --with openpyxl --with scikit-learn==1.9.1 --with imbalanced-learn "
        f"python scripts/run_phase4_imbalance.py --target {args.target} "
        f"--n-splits {args.n_splits} --k {args.k} "
        f"--bootstrap-reps {args.bootstrap_reps} --sensitivity {args.sensitivity}"
    )
    summary = _render_summary(
        target_info=target_info,
        protocol=protocol,
        data=data,
        results=results,
        recommendations=recommendations,
        command=command,
    )

    metrics_path.write_text(
        json.dumps(_sanitise_json(payload), indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(summary, encoding="utf-8")

    print()
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote summary: {summary_path}")
    print(
        "Global best: "
        f"{recommendations['global_best']['model']} / "
        f"{recommendations['global_best']['strategy']} "
        f"(AP={recommendations['global_best']['validation_average_precision']:.3f})"
    )
    print("Test split was not used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
