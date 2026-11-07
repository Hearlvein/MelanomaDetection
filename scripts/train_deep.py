#!/usr/bin/env python3
"""Train a Keras melanoma classifier on the lesion-grouped Phase 1 split.

The test split is never loaded. Use ``scripts/evaluate_deep.py`` for the
one-shot test evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from melanoma_detection import config  # noqa: E402
from melanoma_detection.data import (  # noqa: E402
    load_split_assignments,
    sha256_file,
)
from melanoma_detection.deep import datasets, models, train  # noqa: E402
from melanoma_detection.splits import split_fingerprint  # noqa: E402
from melanoma_detection.training import assert_split_isolation  # noqa: E402


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def _split_counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "validation"):
        subset = frame.loc[frame["split"] == split]
        counts[split] = {
            "rows": int(len(subset)),
            "lesions": int(subset["lesion_id"].nunique()),
            "positive": int(subset["target"].sum()),
            "negative": int(len(subset) - subset["target"].sum()),
        }
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=sorted(config.TARGET_SPECS), default="melanoma")
    parser.add_argument("--model", choices=config.DEEP_MODELS, default="custom_cnn")
    parser.add_argument(
        "--features",
        choices=sorted(config.FEATURE_SPECS),
        default=config.DEFAULT_FEATURE_VERSION,
        help="Only used to locate the split assignment; the CNN does not read tabular features.",
    )
    parser.add_argument("--image-dir", type=Path, default=config.EXTERNAL_DATA_DIR / config.IMAGE_DIR_NAME)
    parser.add_argument("--split-path", type=Path, default=None)
    parser.add_argument("--image-size", type=int, nargs=2, default=list(config.DEEP_IMAGE_SIZE))
    parser.add_argument("--batch-size", type=int, default=config.DEEP_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=config.DEEP_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=config.DEEP_LEARNING_RATE)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--weights", choices=("imagenet", "none"), default="imagenet")
    parser.add_argument("--class-weight", choices=("balanced", "none"), default="balanced")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--no-pointer", action="store_true", help="Do not write models/deep_{target}_frozen.json.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.target not in config.DEEP_TARGETS:
        print(
            f"warning: {args.target!r} is not part of the supported deep line "
            f"({config.DEEP_TARGETS}); results are experimental.",
            file=sys.stderr,
        )

    output_path = args.output or train.deep_model_path(args.target, args.model)
    report_path = args.report or (
        config.METRICS_DIR / f"deep_train_{args.target}_{args.model}.json"
    )
    if output_path.exists() and not args.overwrite:
        print(
            f"error: artifact already exists: {output_path}. Pass --overwrite to retrain.",
            file=sys.stderr,
        )
        return 1

    frame = load_split_assignments(
        args.target, path=args.split_path, feature_version=args.features
    )
    assert_split_isolation(frame)
    train_frame = datasets.split_frame(frame, "train")
    validation_frame = datasets.split_frame(frame, "validation")

    datasets.assert_images_exist(train_frame, args.image_dir)
    datasets.assert_images_exist(validation_frame, args.image_dir)

    image_size = (int(args.image_size[0]), int(args.image_size[1]))
    input_shape = (*image_size, 3)

    train_dataset = datasets.make_tf_dataset(
        train_frame,
        args.image_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        augment=not args.no_augment,
        repeat=True,
    )
    validation_dataset = datasets.make_tf_dataset(
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

    class_weight = train.class_weight_for(train_frame, args.class_weight)
    steps_per_epoch = math.ceil(len(train_frame) / args.batch_size)
    history = train.train_model(
        model,
        train_dataset,
        validation_dataset,
        epochs=args.epochs,
        class_weight=class_weight,
        callbacks=train.default_callbacks(patience=args.patience),
        steps_per_epoch=steps_per_epoch,
    )
    history_dict = train.history_to_dict(history)

    fingerprint = split_fingerprint(frame, args.target, args.seed)
    metadata = {
        "phase": "deep_training",
        "target": args.target,
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
        "feature_version": args.features,
        "split_fingerprint": fingerprint,
        "counts": _split_counts(frame),
        "history": history_dict,
        "history_summary": train.history_summary(history_dict),
        "inputs": {
            "image_dir": _relative(args.image_dir),
            "split_assignments": _relative(
                args.split_path
                or config.split_assignment_path(args.target, args.features)
            ),
        },
        "software": {
            "python": sys.version.split()[0],
            "tensorflow": _tensorflow_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "command": (
            "uv run --python 3.13 --with pandas --with scikit-learn==1.9.1 "
            "--with tensorflow-cpu "
            f"python scripts/train_deep.py --target {args.target} --model {args.model} "
            f"--epochs {args.epochs} --batch-size {args.batch_size} "
            f"--image-size {image_size[0]} {image_size[1]} --weights {args.weights}"
        ),
    }

    artifact_path, sidecar_path = train.save_deep_artifact(model, metadata, output_path)
    pointer_path = None
    if not args.no_pointer:
        pointer_path = train.write_pointer(args.target, args.model, artifact_path, metadata)

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


def _tensorflow_version() -> str:
    import tensorflow as tf

    return tf.__version__


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
