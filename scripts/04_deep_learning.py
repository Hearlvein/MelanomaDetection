#!/usr/bin/env python3
"""Step 4: deep-learning commands for the melanoma task.

Subcommands:
    check-images   Verify that the HAM10000 image files are present.
    train          Train ``custom_cnn`` or ``vgg16`` on the grouped split.
    evaluate       One-shot test evaluation of a trained deep model.

The test split is never loaded by ``train``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from melanoma_detection import config  # noqa: E402
from melanoma_detection.data import (  # noqa: E402
    load_split_assignments,
    sha256_file,
    split_fingerprint,
)
from melanoma_detection.deep import run  # noqa: E402
from melanoma_detection.experiment import assert_split_isolation  # noqa: E402


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def _counts(frame: pd.DataFrame, splits: tuple[str, ...]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for split in splits:
        subset = frame.loc[frame["split"] == split]
        counts[split] = {
            "rows": int(len(subset)),
            "lesions": int(subset["lesion_id"].nunique()),
            "positive": int(subset["target"].sum()),
            "negative": int(len(subset) - subset["target"].sum()),
        }
    return counts


def check_images(args: argparse.Namespace) -> int:
    frame = load_split_assignments()
    splits = ("train", "validation", "test") if args.include_test else ("train", "validation")
    failed = False
    for split in splits:
        subset = run.split_frame(frame, split)
        missing = run.missing_images(subset, args.image_dir)
        print(
            f"{split}: {len(subset)} images, {len(missing)} missing "
            f"under {args.image_dir}"
        )
        if missing:
            print(f"  examples: {missing[: args.examples]}")
            failed = True
    return 1 if failed else 0


def train(args: argparse.Namespace) -> int:
    import tensorflow as tf

    from melanoma_detection.deep import models

    output_path = args.output or config.deep_model_path(args.model)
    report_path = args.report or run.deep_train_report_path(args.model)
    if output_path.exists() and not args.overwrite:
        print(
            f"error: artifact already exists: {output_path}. "
            "Pass --overwrite to retrain.",
            file=sys.stderr,
        )
        return 1

    frame = load_split_assignments()
    assert_split_isolation(frame)
    train_frame = run.split_frame(frame, "train")
    validation_frame = run.split_frame(frame, "validation")
    run.assert_images_exist(train_frame, args.image_dir)
    run.assert_images_exist(validation_frame, args.image_dir)

    image_size = (int(args.image_size[0]), int(args.image_size[1]))
    input_shape = (*image_size, 3)

    train_dataset = run.make_tf_dataset(
        train_frame,
        args.image_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        augment=not args.no_augment,
        repeat=True,
    )
    validation_dataset = run.make_tf_dataset(
        validation_frame,
        args.image_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        shuffle=False,
    )

    weights = None if args.weights == "none" else args.weights
    model = models.build_model(
        args.model,
        input_shape=input_shape,
        weights=weights,
        dropout=args.dropout,
    )
    models.compile_model(model, learning_rate=args.learning_rate)
    model.summary()

    class_weight = run.class_weight_for(train_frame, args.class_weight)
    steps_per_epoch = math.ceil(len(train_frame) / args.batch_size)
    history = run.train_model(
        model,
        train_dataset,
        validation_dataset,
        epochs=args.epochs,
        class_weight=class_weight,
        callbacks=run.default_callbacks(patience=args.patience),
        steps_per_epoch=steps_per_epoch,
    )
    history_dict = run.history_to_dict(history)

    fingerprint = split_fingerprint(frame, config.TARGET_DEFINITION, args.seed)
    metadata = {
        "stage": "deep_training",
        "target": config.TARGET_DEFINITION,
        "model": args.model,
        "model_name": model.name,
        "image_size": list(image_size),
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "learning_rate": args.learning_rate,
        "dropout": args.dropout,
        "weights": args.weights,
        "class_weight_mode": args.class_weight,
        "class_weight": class_weight,
        "augmentation": not args.no_augment,
        "seed": args.seed,
        "split_fingerprint": fingerprint,
        "counts": _counts(frame, ("train", "validation")),
        "history": history_dict,
        "history_summary": run.history_summary(history_dict),
        "inputs": {
            "image_dir": _relative(args.image_dir),
            "split_assignments": _relative(config.split_assignment_path()),
        },
        "software": {
            "python": sys.version.split()[0],
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "command": (
            "uv run --python 3.13 --with pandas --with scikit-learn==1.9.1 "
            "--with tensorflow-cpu python scripts/04_deep_learning.py train "
            f"--model {args.model} --epochs {args.epochs} "
            f"--batch-size {args.batch_size} "
            f"--image-size {image_size[0]} {image_size[1]} --weights {args.weights}"
        ),
    }

    artifact_path, sidecar_path = run.save_deep_artifact(model, metadata, output_path)
    pointer_path = None
    if not args.no_pointer:
        pointer_path = run.write_pointer(args.model, artifact_path, metadata)

    report = dict(metadata)
    report["artifact"] = _relative(artifact_path)
    report["artifact_sha256"] = sha256_file(artifact_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )

    print(f"Wrote model: {artifact_path}")
    print(f"Wrote sidecar: {sidecar_path}")
    if pointer_path is not None:
        print(f"Wrote pointer: {pointer_path}")
    print(f"Wrote training report: {report_path}")
    return 0


def evaluate(args: argparse.Namespace) -> int:
    try:
        run.assert_deep_test_not_yet_evaluated(
            args.model, force=args.force, metrics_path=args.metrics_path
        )
    except FileExistsError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    metrics_path, markdown_path = run.deep_test_paths(
        args.model,
        metrics_path=args.metrics_path,
        markdown_path=args.report_path,
    )

    artifact_path = run.resolve_deep_model_path(args.artifact)
    model, metadata = run.load_deep_artifact(artifact_path)

    image_size = (
        tuple(int(value) for value in args.image_size)
        if args.image_size is not None
        else tuple(int(value) for value in metadata["image_size"])
    )

    frame = load_split_assignments()
    validation_frame = run.split_frame(frame, "validation")
    test_frame = run.split_frame(frame, "test")

    validation_dataset = run.make_tf_dataset(
        validation_frame,
        args.image_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        shuffle=False,
    )
    validation_scores = run.predict_scores(model, validation_dataset)
    thresholds = run.select_thresholds(
        validation_frame["target"].to_numpy(),
        validation_scores,
        min_sensitivity=args.min_sensitivity,
    )
    validation_evaluation = run.evaluate_frame_scores(
        validation_frame,
        validation_scores,
        thresholds["f2_threshold"],
        bootstrap_reps=args.bootstrap_reps,
        seed=config.SEED,
    )

    test_dataset = run.make_tf_dataset(
        test_frame,
        args.image_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_scores = run.predict_scores(model, test_dataset)
    test_evaluation = run.evaluate_frame_scores(
        test_frame,
        test_scores,
        thresholds["f2_threshold"],
        bootstrap_reps=args.bootstrap_reps,
        seed=config.SEED,
    )

    test_at_sensitivity = None
    if thresholds["sensitivity_threshold"] is not None:
        test_at_sensitivity = run.evaluate_frame_scores(
            test_frame,
            test_scores,
            thresholds["sensitivity_threshold"],
            bootstrap_reps=args.bootstrap_reps,
            seed=config.SEED,
        )

    report = {
        "stage": "deep_final_test",
        "target": config.TARGET_DEFINITION,
        "model": args.model,
        "artifact": _relative(artifact_path),
        "artifact_sha256": metadata.get("artifact_sha256"),
        "image_size": list(image_size),
        "split_fingerprint": split_fingerprint(
            frame, config.TARGET_DEFINITION, int(metadata.get("seed", config.SEED))
        ),
        "thresholds": thresholds,
        "validation": validation_evaluation,
        "test": test_evaluation,
        "test_at_sensitivity_threshold": test_at_sensitivity,
        "counts": _counts(frame, ("validation", "test")),
        "command": (
            "uv run --python 3.13 --with pandas --with scikit-learn==1.9.1 "
            "--with tensorflow-cpu python scripts/04_deep_learning.py evaluate "
            f"--model {args.model}"
        ),
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(report, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(_render_deep_summary(report), encoding="utf-8")

    print(f"Wrote test metrics: {metrics_path}")
    print(f"Wrote test report: {markdown_path}")
    return 0


def _render_deep_summary(report: dict[str, object]) -> str:
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
        "The test split was evaluated once. Do not rerun without an explicit,",
        "documented `--force` decision.",
        "",
    ]
    return "\n".join(lines)


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check-images", help="Check image availability.")
    check.add_argument(
        "--image-dir", type=Path, default=config.EXTERNAL_DATA_DIR / config.IMAGE_DIR_NAME
    )
    check.add_argument("--include-test", action="store_true")
    check.add_argument("--examples", type=int, default=5)
    check.set_defaults(func=check_images)

    train_parser = subparsers.add_parser("train", help="Train a deep model.")
    train_parser.add_argument("--model", choices=config.DEEP_MODELS, default="custom_cnn")
    train_parser.add_argument(
        "--image-dir", type=Path, default=config.EXTERNAL_DATA_DIR / config.IMAGE_DIR_NAME
    )
    train_parser.add_argument(
        "--image-size", type=int, nargs=2, default=list(config.DEEP_IMAGE_SIZE)
    )
    train_parser.add_argument("--batch-size", type=int, default=config.DEEP_BATCH_SIZE)
    train_parser.add_argument("--epochs", type=int, default=config.DEEP_EPOCHS)
    train_parser.add_argument(
        "--learning-rate", type=float, default=config.DEEP_LEARNING_RATE
    )
    train_parser.add_argument("--dropout", type=float, default=0.5)
    train_parser.add_argument("--weights", choices=("imagenet", "none"), default="imagenet")
    train_parser.add_argument(
        "--class-weight", choices=("balanced", "none"), default="balanced"
    )
    train_parser.add_argument("--patience", type=int, default=8)
    train_parser.add_argument("--seed", type=int, default=config.SEED)
    train_parser.add_argument("--no-augment", action="store_true")
    train_parser.add_argument("--output", type=Path, default=None)
    train_parser.add_argument("--report", type=Path, default=None)
    train_parser.add_argument("--no-pointer", action="store_true")
    train_parser.add_argument("--overwrite", action="store_true")
    train_parser.set_defaults(func=train)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="One-shot deep test evaluation."
    )
    evaluate_parser.add_argument("--model", choices=config.DEEP_MODELS, default="custom_cnn")
    evaluate_parser.add_argument("--artifact", type=Path, default=None)
    evaluate_parser.add_argument(
        "--image-dir", type=Path, default=config.EXTERNAL_DATA_DIR / config.IMAGE_DIR_NAME
    )
    evaluate_parser.add_argument("--metrics-path", type=Path, default=None)
    evaluate_parser.add_argument("--report-path", type=Path, default=None)
    evaluate_parser.add_argument("--image-size", type=int, nargs=2, default=None)
    evaluate_parser.add_argument("--batch-size", type=int, default=config.DEEP_BATCH_SIZE)
    evaluate_parser.add_argument("--min-sensitivity", type=float, default=0.90)
    evaluate_parser.add_argument("--bootstrap-reps", type=int, default=2_000)
    evaluate_parser.add_argument("--seed", type=int, default=config.SEED)
    evaluate_parser.add_argument("--force", action="store_true")
    evaluate_parser.set_defaults(func=evaluate)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
