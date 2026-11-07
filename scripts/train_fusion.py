#!/usr/bin/env python3
"""Train the tabular + image-embedding late-fusion model.

The fusion model uses V2 handcrafted features plus ``emb_*`` columns from
``scripts/extract_embeddings.py``. All preprocessing, embedding scaling, and PCA
are fitted inside the training split. The validation split is used once for the
reported validation metrics; the test split is never scored here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from melanoma_detection import config  # noqa: E402
from melanoma_detection.data import load_modeling_frame, sha256_file  # noqa: E402
from melanoma_detection.fusion.embeddings import (  # noqa: E402
    embedding_feature_names,
    embeddings_path,
    load_embeddings,
    merge_embeddings,
)
from melanoma_detection.fusion.pipeline import fusion_artifact_path  # noqa: E402
from melanoma_detection.fusion.train import (  # noqa: E402
    fit_fusion,
    save_fusion_artifact,
    write_fusion_pointer,
)
from melanoma_detection.splits import split_fingerprint  # noqa: E402
from melanoma_detection.training import assert_split_isolation  # noqa: E402


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


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


def _split_counts(frame) -> dict[str, dict[str, int]]:
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


def _render_markdown(report: dict[str, Any]) -> str:
    validation = report["validation"]["metrics"]
    high = report.get("validation_at_sensitivity_threshold")
    lines = [
        f"# Fusion training — {report['target']}",
        "",
        f"- Model: **{report['model']}**",
        f"- Feature version: `{report['feature_version']}`",
        f"- Embedding model: `{report['embedding_model']}`",
        f"- PCA components: `{report['n_components']}`",
        f"- Artifact: `{report['artifact']}`",
        f"- Split fingerprint: `{report['split_fingerprint']}`",
        f"- F2 threshold (grouped OOF): `{report['threshold_selection']['f2_threshold']:.6f}`",
        f"- Sensitivity threshold: `{report['threshold_selection']['sensitivity_threshold']}`",
        "",
        "## Out-of-fold and validation scores",
        "",
        "| Metric | Grouped OOF | Validation |",
        "| --- | ---: | ---: |",
        f"| Average precision | {report['oof']['average_precision']:.4f} | {validation['average_precision']:.4f} |",
        f"| ROC AUC | {report['oof']['roc_auc']:.4f} | {validation['roc_auc']:.4f} |",
        f"| Sensitivity | — | {validation['sensitivity']:.4f} |",
        f"| Specificity | — | {validation['specificity']:.4f} |",
        f"| F2 | {report['oof']['f2']:.4f} | {validation['f2']:.4f} |",
        "",
    ]
    if high is not None:
        lines.extend(
            [
                "## High-sensitivity validation operating point",
                "",
                "| Sensitivity | Specificity | PPV | NPV | F2 |",
                "| ---: | ---: | ---: | ---: | ---: |",
                f"| {high['sensitivity']:.4f} | {high['specificity']:.4f} | "
                f"{high['ppv']:.4f} | {high['npv']:.4f} | {high['f2']:.4f} |",
                "",
            ]
        )
    lines.extend(
        [
            "## Note",
            "",
            "The threshold was selected from grouped out-of-fold scores inside the",
            "training split. Validation was used once for this report. The test split",
            "was not scored; use `scripts/evaluate_fusion.py` for the one-shot test.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=sorted(config.TARGET_SPECS), default="melanoma")
    parser.add_argument(
        "--features",
        choices=sorted(config.FEATURE_SPECS),
        default="v2",
        help="Tabular feature version. Fusion is defined on V2.",
    )
    parser.add_argument("--model", choices=config.DEEP_MODELS, default=config.FUSION_EMBEDDING_MODEL)
    parser.add_argument("--embeddings", type=Path, default=None)
    parser.add_argument("--interim-path", type=Path, default=None)
    parser.add_argument("--split-path", type=Path, default=None)
    parser.add_argument("--n-components", type=int, default=config.FUSION_N_COMPONENTS)
    parser.add_argument("--n-splits", type=int, default=config.N_SPLITS)
    parser.add_argument("--bootstrap-reps", type=int, default=2_000)
    parser.add_argument("--min-sensitivity", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--no-pointer", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.target not in config.FUSION_TARGETS:
        print(
            f"warning: {args.target!r} is not part of the supported fusion line "
            f"({config.FUSION_TARGETS}); results are experimental.",
            file=sys.stderr,
        )

    output_path = args.output or fusion_artifact_path(args.target)
    report_path = args.report or (
        config.METRICS_DIR / f"fusion_train_{args.target}.json"
    )
    markdown_path = report_path.with_suffix(".md")
    if output_path.exists() and not args.overwrite:
        print(
            f"error: artifact already exists: {output_path}. Pass --overwrite to retrain.",
            file=sys.stderr,
        )
        return 1

    embeddings_file = args.embeddings or embeddings_path(args.model, args.target)
    image_ids, vectors = load_embeddings(embeddings_file)
    embedding_columns = list(embedding_feature_names(vectors.shape[1]))

    frame = load_modeling_frame(
        args.target,
        interim_path=args.interim_path,
        split_path=args.split_path,
        feature_version=args.features,
    )
    assert_split_isolation(frame)
    fingerprint = split_fingerprint(frame, args.target, args.seed)
    counts = _split_counts(frame)

    # Only the training and validation splits are needed to fit and report.
    frame = frame.loc[frame["split"].isin(("train", "validation"))].reset_index(drop=True)
    frame = merge_embeddings(frame, image_ids, vectors)

    numeric_features = list(config.get_numeric_features(args.features))
    categorical_features = list(config.get_categorical_features(args.features))

    pipeline, result, feature_columns = fit_fusion(
        frame,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        embedding_features=embedding_columns,
        n_components=args.n_components,
        n_splits=args.n_splits,
        seed=args.seed,
        bootstrap_reps=args.bootstrap_reps,
        min_sensitivity=args.min_sensitivity,
    )

    metadata: dict[str, Any] = {
        "phase": "fusion_training",
        "target": args.target,
        "model": config.FUSION_MODEL_NAME,
        "feature_version": args.features,
        "embedding_model": args.model,
        "n_components": args.n_components,
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "embedding_features": embedding_columns,
        "split_fingerprint": fingerprint,
        "counts": counts,
        "seed": args.seed,
        "inputs": {
            "interim_dataset": _relative(
                args.interim_path or config.interim_dataset_path(args.features)
            ),
            "split_assignments": _relative(
                args.split_path
                or config.split_assignment_path(args.target, args.features)
            ),
            "embeddings": _relative(embeddings_file),
            "embeddings_sha256": sha256_file(embeddings_file),
        },
        **{key: value for key, value in result.items() if key != "embedding_features"},
        "command": (
            "uv run --python 3.13 --with pandas --with scikit-learn==1.9.1 "
            "--with numpy --with joblib "
            f"python scripts/train_fusion.py --target {args.target} "
            f"--features {args.features} --model {args.model} "
            f"--n-components {args.n_components}"
        ),
    }

    artifact_path, sidecar_path = save_fusion_artifact(pipeline, metadata, output_path)
    pointer_path = None
    if not args.no_pointer:
        pointer_path = write_fusion_pointer(args.target, artifact_path, metadata)

    report = dict(metadata)
    report["artifact"] = _relative(artifact_path)
    report["artifact_sha256"] = sha256_file(artifact_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")

    print(f"Wrote model: {artifact_path}")
    print(f"Wrote sidecar: {sidecar_path}")
    if pointer_path is not None:
        print(f"Wrote pointer: {pointer_path}")
    print(f"Wrote training report: {report_path}")
    print(f"Wrote training summary: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
