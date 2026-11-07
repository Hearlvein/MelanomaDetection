#!/usr/bin/env python3
"""Phase 5: one-shot evaluation of the frozen model on the test split.

This script loads a frozen artifact, verifies that it belongs to the current
target and split, then evaluates the test split exactly once. It refuses to
overwrite an existing final-test report unless ``--force`` is passed.
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
from melanoma_detection.data import load_modeling_frame  # noqa: E402
from melanoma_detection.evaluate import binary_metrics  # noqa: E402
from melanoma_detection.splits import split_fingerprint  # noqa: E402
from melanoma_detection.training import (  # noqa: E402
    assert_split_isolation,
    evaluate_at_threshold,
    positive_scores,
    split_counts,
)
from melanoma_detection.train import (  # noqa: E402
    assert_test_not_yet_evaluated,
    final_test_paths,
    load_artifact,
    resolve_artifact_path,
)

BINARY_TARGETS = ("melanoma", "malignant")


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


def _format_value(value: float | int | None) -> str:
    if value is None:
        return "—"
    number = float(value)
    return f"{number:.3f}" if np.isfinite(number) else "—"


def _format_with_ci(
    metrics: dict[str, Any],
    intervals: dict[str, dict[str, float]],
    metric: str,
) -> str:
    value = _format_value(metrics.get(metric))
    interval = intervals.get(metric)
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


def _write_figures(
    *,
    target: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    output_dir: Path,
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve
        from sklearn.metrics import precision_recall_curve, roc_curve
    except ImportError:
        print("matplotlib is not installed; skipping figures.")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    prevalence = float(np.mean(y_true))

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = float(binary_metrics(y_true, y_score, threshold)["average_precision"])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"AP = {ap:.3f}")
    ax.axhline(prevalence, color="grey", linestyle="--", label=f"Prevalence = {prevalence:.3f}")
    ax.set_xlabel("Recall / sensitivity")
    ax.set_ylabel("Precision / PPV")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Phase 5 final test precision-recall ({target})")
    ax.legend(loc="best")
    fig.tight_layout()
    path = output_dir / f"phase5_final_test_{target}_precision_recall.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(_relative(path))

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = float(binary_metrics(y_true, y_score, threshold)["roc_auc"])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Chance")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate / sensitivity")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Phase 5 final test ROC ({target})")
    ax.legend(loc="best")
    fig.tight_layout()
    path = output_dir / f"phase5_final_test_{target}_roc.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(_relative(path))

    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Phase 5 final test calibration ({target})")
    ax.legend(loc="best")
    fig.tight_layout()
    path = output_dir / f"phase5_final_test_{target}_calibration.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(_relative(path))

    cm = binary_metrics(y_true, y_score, threshold)["confusion_matrix"]
    matrix = np.asarray([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1], labels=["Predicted 0", "Predicted 1"])
    ax.set_yticks([0, 1], labels=["Actual 0", "Actual 1"])
    for row in range(2):
        for column in range(2):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center")
    ax.set_title(f"Phase 5 final test confusion matrix ({target})")
    fig.tight_layout()
    path = output_dir / f"phase5_final_test_{target}_confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(_relative(path))

    return written


def _render_summary(
    *,
    target_info: dict[str, Any],
    metadata: dict[str, Any],
    artifact: Path,
    data: dict[str, Any],
    evaluation: dict[str, Any],
    high_sensitivity: dict[str, Any] | None,
    figures: list[str],
    command: str,
) -> str:
    metrics = evaluation["metrics"]
    intervals = evaluation["confidence_intervals"]
    validation = metadata["validation"]["metrics"]
    threshold = metadata["threshold"]["f2_threshold"]

    comparison_rows = [
        [
            "Average precision",
            _format_value(validation.get("average_precision")),
            _format_with_ci(metrics, intervals, "average_precision"),
        ],
        [
            "ROC-AUC",
            _format_value(validation.get("roc_auc")),
            _format_with_ci(metrics, intervals, "roc_auc"),
        ],
        [
            "Sensitivity",
            _format_value(validation.get("sensitivity")),
            _format_with_ci(metrics, intervals, "sensitivity"),
        ],
        [
            "Specificity",
            _format_value(validation.get("specificity")),
            _format_with_ci(metrics, intervals, "specificity"),
        ],
        [
            "PPV",
            _format_value(validation.get("ppv")),
            _format_with_ci(metrics, intervals, "ppv"),
        ],
        [
            "F2",
            _format_value(validation.get("f2")),
            _format_with_ci(metrics, intervals, "f2"),
        ],
        [
            "Brier score",
            _format_value(validation.get("brier_score")),
            _format_with_ci(metrics, intervals, "brier_score"),
        ],
    ]

    lines = [
        "# Phase 5 — Final test evaluation",
        "",
        f"- Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Target: **{target_info['definition']}**",
        f"- Positive class: **{target_info['labels'][1]}** (`target = 1`)",
        f"- Negative class: **{target_info['labels'][0]}** (`target = 0`)",
        f"- Model: **{metadata['model']['label']}**",
        f"- Imbalance strategy: `{metadata['imbalance_strategy']}`",
        f"- Frozen F2 threshold: {_format_value(threshold)}",
        f"- Frozen artifact: `{_relative(artifact)}`",
        f"- Split fingerprint: `{metadata['split']['fingerprint']}`",
        "",
        "> This report evaluates the test split **once**. The model and threshold were",
        "> frozen before this script ran. Do not rerun without an explicit `--force`.",
        "",
        "## Test data",
        "",
        _markdown_table(
            ["Rows", "Lesions", "Positive", "Negative", "Prevalence"],
            [
                [
                    data["rows"],
                    data["lesions"],
                    data["positive"],
                    data["negative"],
                    _format_value(data["prevalence"]),
                ]
            ],
        ),
        "",
        "## Test metrics at the frozen threshold",
        "",
        _markdown_table(
            [
                "Sensitivity",
                "Specificity",
                "PPV",
                "NPV",
                "F1",
                "F2",
                "AP",
                "ROC-AUC",
                "Brier",
                "Predicted positive",
            ],
            [
                [
                    _format_with_ci(metrics, intervals, "sensitivity"),
                    _format_with_ci(metrics, intervals, "specificity"),
                    _format_with_ci(metrics, intervals, "ppv"),
                    _format_with_ci(metrics, intervals, "npv"),
                    _format_with_ci(metrics, intervals, "f1"),
                    _format_with_ci(metrics, intervals, "f2"),
                    _format_with_ci(metrics, intervals, "average_precision"),
                    _format_with_ci(metrics, intervals, "roc_auc"),
                    _format_with_ci(metrics, intervals, "brier_score"),
                    _format_value(metrics["predicted_positive_rate"]),
                ]
            ],
        ),
        "",
        "## Confusion matrix",
        "",
        _markdown_table(
            ["TP", "FP", "FN", "TN"],
            [
                [
                    evaluation["confusion_matrix"]["tp"],
                    evaluation["confusion_matrix"]["fp"],
                    evaluation["confusion_matrix"]["fn"],
                    evaluation["confusion_matrix"]["tn"],
                ]
            ],
        ),
        "",
        "## Validation vs test",
        "",
        "Validation was the model-selection split, so its estimate is mildly optimistic.",
        "The test estimate below is the one to report as the unbiased estimate.",
        "",
        _markdown_table(
            ["Metric", "Validation", "Test (95% CI)"],
            comparison_rows,
        ),
        "",
    ]

    if high_sensitivity is not None:
        lines.extend(
            [
                "## High-sensitivity operating point",
                "",
                _markdown_table(
                    ["Threshold", "Sensitivity", "Specificity", "PPV", "NPV", "Predicted positive"],
                    [
                        [
                            _format_value(metadata["threshold"]["sensitivity_threshold"]),
                            _format_value(high_sensitivity["sensitivity"]),
                            _format_value(high_sensitivity["specificity"]),
                            _format_value(high_sensitivity["ppv"]),
                            _format_value(high_sensitivity["npv"]),
                            _format_value(high_sensitivity["predicted_positive_rate"]),
                        ]
                    ],
                ),
                "",
            ]
        )

    if figures:
        lines.extend(
            [
                "## Figures",
                "",
                *[f"- `{path}`" for path in figures],
                "",
            ]
        )

    lines.extend(
        [
            "## Model card summary",
            "",
            f"- **Intended use:** research evaluation of {target_info['definition']} detection on HAM10000 tabular features.",
            "- **Not for clinical use:** this is not a diagnostic device and must not guide patient care.",
            f"- **Target definition:** {target_info['description']}",
            f"- **Features:** {len(metadata['feature_columns'])} tabular metadata, colour, and shape features.",
            f"- **Training rows:** {metadata['training']['rows']} images from {metadata['training']['lesions']} lesions.",
            f"- **Validation rows:** {metadata['validation_data']['rows']} images from {metadata['validation_data']['lesions']} lesions.",
            "- **Grouping:** lesions are disjoint across train/validation/test; confidence intervals resample lesions.",
            "- **Threshold:** selected from grouped-CV out-of-fold predictions on the training split.",
            "",
            "## Limitations",
            "",
            "- HAM10000 is a dermatoscopic image dataset; the model uses pre-extracted tabular features, not raw pixels.",
            "- No patient identifier is available, so lesion grouping is the strongest available leakage control.",
            "- The dataset is not representative of all skin tones, populations, or clinical workflows.",
            "- Validation was used for model selection; test metrics are the unbiased estimate but still reflect one dataset.",
            "- Probability calibration may be imperfect, especially after class weighting or resampling.",
            "- Performance at this operating point is not sufficient for clinical deployment.",
            "",
            "## Reproduce",
            "",
            "```bash",
            command,
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=BINARY_TARGETS, default=config.TARGET_DEFINITION)
    parser.add_argument(
        "--artifact",
        default=None,
        help="Path to a frozen joblib artifact. Defaults to the target's frozen pointer.",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=2_000)
    parser.add_argument("--figures", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing final-test report. Use only intentionally.",
    )
    args = parser.parse_args()

    # Check the one-shot guard before doing any test work.
    metrics_path, summary_path = final_test_paths(args.target)
    assert_test_not_yet_evaluated(args.target, force=args.force)

    frame = load_modeling_frame(args.target)
    assert_split_isolation(frame)

    artifact = resolve_artifact_path(args.target, args.artifact)
    pipeline, metadata = load_artifact(artifact)

    artifact_target = metadata.get("target", {}).get("definition")
    if artifact_target != args.target:
        raise RuntimeError(
            f"artifact target is {artifact_target!r}, but --target is {args.target!r}"
        )

    current_fingerprint = split_fingerprint(frame, args.target, config.SEED)
    frozen_fingerprint = metadata.get("split", {}).get("fingerprint")
    if frozen_fingerprint != current_fingerprint:
        raise RuntimeError(
            "split fingerprint mismatch: the frozen model was trained on a different "
            f"split ({frozen_fingerprint!r} != {current_fingerprint!r})"
        )

    test = frame.loc[frame["split"] == "test"].reset_index(drop=True)
    if test.empty:
        raise RuntimeError("test split is empty")

    feature_columns = metadata["feature_columns"]
    y_test = test["target"].to_numpy()
    groups_test = test[config.GROUP_COLUMN].to_numpy()
    test_scores = positive_scores(pipeline.predict_proba(test[feature_columns]))

    threshold = float(metadata["threshold"]["f2_threshold"])
    evaluation = evaluate_at_threshold(
        y_test,
        test_scores,
        groups_test,
        threshold=threshold,
        bootstrap_reps=args.bootstrap_reps,
        seed=config.SEED,
    )

    sensitivity_threshold = metadata["threshold"].get("sensitivity_threshold")
    high_sensitivity = None
    if sensitivity_threshold is not None:
        high_sensitivity = binary_metrics(
            y_test, test_scores, threshold=float(sensitivity_threshold)
        )

    figures: list[str] = []
    if args.figures:
        figures = _write_figures(
            target=args.target,
            y_true=y_test,
            y_score=test_scores,
            threshold=threshold,
            output_dir=config.REPORTS_DIR / "figures",
        )

    target_info = metadata["target"]
    # JSON object keys are always strings, so normalise the label mapping back
    # to integer class labels before rendering the report.
    target_info["labels"] = {
        int(label): name for label, name in target_info.get("labels", {}).items()
    }
    data = split_counts(frame)["test"]
    command = (
        "uv run --python 3.13 "
        "--with pandas --with openpyxl --with scikit-learn==1.9.1 "
        "--with imbalanced-learn --with matplotlib "
        f"python scripts/evaluate_final_test.py --target {args.target} "
        f"--artifact {_relative(artifact)} --bootstrap-reps {args.bootstrap_reps}"
    )

    payload = {
        "phase": 5,
        "stage": "final_test",
        "target": target_info,
        "model": metadata["model"],
        "imbalance_strategy": metadata["imbalance_strategy"],
        "artifact": {
            "path": _relative(artifact),
            "sha256": metadata.get("artifact_sha256"),
        },
        "threshold": metadata["threshold"],
        "split_fingerprint": frozen_fingerprint,
        "one_shot": True,
        "test": {
            "data": data,
            "metrics": evaluation["metrics"],
            "confidence_intervals": evaluation["confidence_intervals"],
            "confusion_matrix": evaluation["confusion_matrix"],
        },
        "test_at_sensitivity_threshold": high_sensitivity,
        "validation_reference": metadata["validation"],
        "comparison": {
            "validation_average_precision": metadata["validation"]["metrics"].get(
                "average_precision"
            ),
            "test_average_precision": evaluation["metrics"].get("average_precision"),
            "validation_roc_auc": metadata["validation"]["metrics"].get("roc_auc"),
            "test_roc_auc": evaluation["metrics"].get("roc_auc"),
        },
        "figures": figures,
        "software": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "command": command,
        "notes": [
            "The test split was evaluated once after freezing the model and threshold.",
            "Confidence intervals resample whole lesions, not individual images.",
            "Not clinical evidence; research use only.",
        ],
    }

    summary = _render_summary(
        target_info=target_info,
        metadata=metadata,
        artifact=artifact,
        data=data,
        evaluation=evaluation,
        high_sensitivity=high_sensitivity,
        figures=figures,
        command=command,
    )

    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(_sanitise_json(payload), indent=2) + "\n", encoding="utf-8"
    )
    summary_path.write_text(summary, encoding="utf-8")

    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote summary: {summary_path}")
    print(
        "Test: "
        f"sensitivity={evaluation['metrics']['sensitivity']:.3f}, "
        f"specificity={evaluation['metrics']['specificity']:.3f}, "
        f"PPV={evaluation['metrics']['ppv']:.3f}, "
        f"AP={evaluation['metrics']['average_precision']:.3f}, "
        f"ROC-AUC={evaluation['metrics']['roc_auc']:.3f}"
    )
    print("The test split has now been evaluated once.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
