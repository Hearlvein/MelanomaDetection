#!/usr/bin/env python3
"""Extract image embeddings for late fusion.

Two modes are supported:

- ``--artifact``: use the ``embedding`` layer of a trained deep model.
- default: use frozen pretrained VGG16 global-average-pooled features.

The embedding table stores one row per ``image_id`` and a JSON manifest with
hashes, dimensions, and the exact command. Embeddings are label-free features;
models and thresholds are still fitted on the training split only.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from melanoma_detection import config  # noqa: E402
from melanoma_detection.data import load_split_assignments  # noqa: E402
from melanoma_detection.deep import embeddings, models  # noqa: E402
from melanoma_detection.deep.train import load_deep_artifact  # noqa: E402
from melanoma_detection.fusion.embeddings import (  # noqa: E402
    embeddings_manifest_path,
    embeddings_path,
    save_embeddings,
)
from melanoma_detection.splits import split_fingerprint  # noqa: E402
from melanoma_detection.training import assert_split_isolation, split_counts  # noqa: E402


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=sorted(config.TARGET_SPECS), default="melanoma")
    parser.add_argument(
        "--features",
        choices=sorted(config.FEATURE_SPECS),
        default="v2",
        help="Feature version used to locate the split assignment.",
    )
    parser.add_argument("--model", choices=config.DEEP_MODELS, default=config.FUSION_EMBEDDING_MODEL)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=None,
        help="Trained .keras model. Omit to use pretrained VGG16 features.",
    )
    parser.add_argument("--layer", default="embedding")
    parser.add_argument("--image-dir", type=Path, default=config.EXTERNAL_DATA_DIR / config.IMAGE_DIR_NAME)
    parser.add_argument("--split-path", type=Path, default=None)
    parser.add_argument("--image-size", type=int, nargs=2, default=None)
    parser.add_argument("--batch-size", type=int, default=config.DEEP_BATCH_SIZE)
    parser.add_argument("--weights", choices=("imagenet", "none"), default="imagenet")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.target not in config.FUSION_TARGETS:
        print(
            f"warning: {args.target!r} is not part of the supported fusion line "
            f"({config.FUSION_TARGETS}); results are experimental.",
            file=sys.stderr,
        )

    output_path = args.output or embeddings_path(args.model, args.target)
    manifest_path = args.manifest or embeddings_manifest_path(args.model, args.target)
    if output_path.exists() and not args.overwrite:
        print(
            f"error: embedding table already exists: {output_path}. "
            "Pass --overwrite to regenerate it.",
            file=sys.stderr,
        )
        return 1

    frame = load_split_assignments(
        args.target, path=args.split_path, feature_version=args.features
    )
    assert_split_isolation(frame)

    artifact_path: Path | None = None
    metadata: dict[str, object] = {}
    if args.artifact is not None:
        artifact_path = (
            args.artifact
            if args.artifact.is_absolute()
            else PROJECT_ROOT / args.artifact
        )
        model, metadata = load_deep_artifact(artifact_path)
        image_size = (
            tuple(int(value) for value in args.image_size)
            if args.image_size is not None
            else tuple(int(value) for value in metadata["image_size"])
        )
        weights = metadata.get("weights", "unknown")
    else:
        image_size = (
            tuple(int(value) for value in args.image_size)
            if args.image_size is not None
            else config.DEEP_IMAGE_SIZE
        )
        weights = args.weights
        model = models.build_vgg16_embedding_model(
            input_shape=(*image_size, 3),
            weights=None if weights == "none" else weights,
        )

    vectors = embeddings.extract_embeddings(
        model,
        frame,
        args.image_dir,
        layer_name=args.layer,
        image_size=image_size,
        batch_size=args.batch_size,
    )
    if vectors.ndim != 2 or vectors.shape[1] < 1:
        raise RuntimeError(f"unexpected embedding shape: {vectors.shape}")

    command = (
        "uv run --python 3.13 --with pandas --with pillow --with numpy "
        "--with tensorflow-cpu "
        f"python scripts/extract_embeddings.py --target {args.target} "
        f"--model {args.model} --features {args.features} "
        f"--image-size {image_size[0]} {image_size[1]} --weights {weights}"
    )
    extra = {
        "phase": "embedding_extraction",
        "features": args.features,
        "layer": args.layer,
        "image_size": list(image_size),
        "batch_size": args.batch_size,
        "weights": weights,
        "seed": args.seed,
        "split_fingerprint": split_fingerprint(frame, args.target, args.seed),
        "counts": split_counts(frame),
        "inputs": {
            "image_dir": _relative(args.image_dir),
            "split_assignments": _relative(
                args.split_path
                or config.split_assignment_path(args.target, args.features)
            ),
            "deep_artifact": _relative(artifact_path) if artifact_path else None,
            "deep_artifact_sha256": (
                metadata.get("artifact_sha256") if artifact_path else None
            ),
        },
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
        "command": command,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    path, manifest = save_embeddings(
        output_path,
        frame["image_id"].to_numpy(),
        vectors,
        model_name=args.model,
        target=args.target,
        metadata=extra,
        manifest_path=manifest_path,
    )

    print(f"Wrote embeddings: {path} ({vectors.shape[0]:,} rows, {vectors.shape[1]} dims)")
    print(f"Wrote manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
