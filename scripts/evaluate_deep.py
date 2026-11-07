#!/usr/bin/env python3
"""One-shot test evaluation for a trained Keras melanoma classifier.

The F2 threshold and the secondary high-sensitivity threshold are selected on
the validation split, then the test split is scored exactly once.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from melanoma_detection import config  # noqa: E402
from melanoma_detection.data import load_split_assignments  # noqa: E402
from melanoma_detection.deep import datasets, evaluate, train  # noqa: E402
from melanoma_detection.splits import split_fingerprint  # noqa: E402


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def _render_markdown(report: dict[str, Any]) -> str:
    validation = report["validation"]["metrics"]
    test = report["test"]["metrics"]
    high = report.get("test_at_sensitivity_threshold")
    high_metrics = high["metrics"] if high else None

    def row(name: str, key: str) -> str:
        values = [f"{validation[key]:.4f}", f"{test[key]:.4f}"]
        if high_metrics is not None:
            values.append(f"{high_metrics[key]:.4f}")
        return "| " + name + " | " + " | ".join(values) + " |"

    header = "| Metric | Validation | Test |"
    divider = "| --- | ---: | ---: |"
    if high_metrics is not None:
        header = "| Metric | Validation | Test | Test @ sensitivity threshold |"
        divider = "| --- | ---: | ---: | ---: |"

    lines = [
        f"# Deep final test — {report['target']} / {report['model']}",
        "",
        f"- Generated at: `{report['created_at']}`",
        f"- Artifact: `{report['artifact']}`",
        f"- Split fingerprint: `{report['split_fingerprint']}`",
        f"- F2 threshold (selected on validation): `{report['thresholds']['f2_threshold']:.6f}`",
        f"- Sensitivity target: `{report['thresholds']['sensitivity_target']}`",
        f"- Sensitivity threshold: `{report['thresholds']['sensitivity_threshold']}`",
        "",
        "## Scores",
        "",
        header,
        divider,
        row("Average precision", "average_precision"),
        row("ROC AUC", "roc_auc"),
        row("Sensitivity", "sensitivity"),
        row("Specificity", "specificity"),
        row("PPV", "ppv"),
        row("NPV", "npv"),
        row("F1", "f1"),
        row("F2", "f2"),
        row("Accuracy", "accuracy"),
        row("Balanced accuracy", "balanced_accuracy"),
        row("Brier score", "brier_score"),
        row("Predicted positive rate", "predicted_positive_rate"),
        "",
        "## Confusion matrix (F2 threshold)",
        "",
        "| Split | TN | FP | FN | TP |",
        "| --- | ---: | ---: | ---: | ---: |",
        (
            f"| validation | {validation['confusion_matrix']['tn']} | "
            f"{validation['confusion_matrix']['fp']} | "
            f"{validation['confusion_matrix']['fn']} | "
            f"{validation['confusion_matrix']['tp']} |"
        ),
        (
            f"| test | {test['confusion_matrix']['tn']} | "
            f"{test['confusion_matrix']['fp']} | "
            f"{test['confusion_matrix']['fn']} | "
            f"{test['confusion_matrix']['tp']} |"
        ),
        "",
        "## Note",
        "",
        "The test split was evaluated once. Do not rerun without an explicit,",
        "documented `--force` decision.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=sorted(config.TARGET_SPECS), default="melanoma")
    parser.add_argument("--model", choices=config.DEEP_MODELS, default="custom_cnn")
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument("--image-dir", type=Path, default=config.EXTERNAL_DATA_DIR / config.IMAGE_DIR_NAME)
    parser.add_argument("--split-path", type=Path, default=None)
    parser.add_argument("--metrics-path", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--image-size", type=int, nargs=2, default=None)
    parser.add_argument("--batch-size", type=int, default=config.DEEP_BATCH_SIZE)
    parser.add_argument("--min-sensitivity", type=float, default=0.90)
    parser.add_argument("--bootstrap-reps", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing test report.")
    args = parser.parse_args()

    # Guard before touching the test split.
    try:
        evaluate.assert_deep_test_not_yet_evaluated(
            args.target,
            args.model,
            force=args.force,
            metrics_path=args.metrics_path,
        )
    except FileExistsError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    metrics_path, markdown_path = evaluate.deep_test_paths(
        args.target,
        args.model,
        metrics_path=args.metrics_path,
        markdown_path=args.report_path,
    )

    artifact_path = train.resolve_deep_model_path(args.target, args.artifact)
    model, metadata = train.load_deep_artifact(artifact_path)

    image_size = (
        tuple(int(value) for value in args.image_size)
        if args.image_size is not None
        else tuple(int(value) for value in metadata["image_size"])
    )

    frame = load_split_assignments(
        args.target,
        path=args.split_path,
        feature_version=metadata.get("feature_version", "v1"),
    )
    validation_frame = datasets.split_frame(frame, "validation")
    test_frame = datasets.split_frame(frame, "test")

    validation_dataset = datasets.make_tf_dataset(
        validation_frame,
        args.image_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        shuffle=False,
    )
    validation_scores = evaluate.predict_scores(model, validation_dataset)
    thresholds = evaluate.select_thresholds(
        validation_frame["target"].to_numpy(),
        validation_scores,
        min_sensitivity=args.min_sensitivity,
    )

    validation_evaluation = evaluate.evaluate_frame_scores(
        validation_frame,
        validation_scores,
        thresholds["f2_threshold"],
        bootstrap_reps=args.bootstrap_reps,
        seed=args.seed,
    )

    test_dataset = datasets.make_tf_dataset(
        test_frame,
        args.image_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_scores = evaluate.predict_scores(model, test_dataset)
    test_evaluation = evaluate.evaluate_frame_scores(
        test_frame,
        test_scores,
        thresholds["f2_threshold"],
        bootstrap_reps=args.bootstrap_reps,
        seed=args.seed,
    )

    test_at_sensitivity = None
    if thresholds["sensitivity_threshold"] is not None:
        test_at_sensitivity = evaluate.evaluate_frame_scores(
            test_frame,
            test_scores,
            thresholds["sensitivity_threshold"],
            bootstrap_reps=args.bootstrap_reps,
            seed=args.seed,
        )

    report: dict[str, Any] = {
        "phase": "deep_final_test",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": args.target,
        "model": args.model,
        "artifact": _relative(artifact_path),
        "artifact_sha256": metadata.get("artifact_sha256"),
        "image_size": list(image_size),
        "split_fingerprint": split_fingerprint(
            frame, args.target, int(metadata.get("seed", config.SEED))
        ),
        "thresholds": thresholds,
        "validation": validation_evaluation,
        "test": test_evaluation,
        "test_at_sensitivity_threshold": test_at_sensitivity,
        "counts": {
            "validation": {
                "rows": int(len(validation_frame)),
                "lesions": int(validation_frame["lesion_id"].nunique()),
                "positive": int(validation_frame["target"].sum()),
            },
            "test": {
                "rows": int(len(test_frame)),
                "lesions": int(test_frame["lesion_id"].nunique()),
                "positive": int(test_frame["target"].sum()),
            },
        },
        "command": (
            "uv run --python 3.13 --with pandas --with scikit-learn==1.9.1 "
            "--with tensorflow-cpu "
            f"python scripts/evaluate_deep.py --target {args.target} --model {args.model}"
        ),
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(report, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")

    print(f"Wrote test metrics: {metrics_path}")
    print(f"Wrote test report: {markdown_path}")
    print()
    print(_render_markdown(report))
    return 0


def _json_default(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
