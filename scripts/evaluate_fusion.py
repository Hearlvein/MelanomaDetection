#!/usr/bin/env python3
"""One-shot test evaluation for the tabular + embedding fusion model.

The model and thresholds were frozen by ``scripts/train_fusion.py``. This
script scores the validation split for reference, then the test split exactly
once. It refuses to overwrite an existing report without ``--force``.
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
from melanoma_detection.data import load_modeling_frame, sha256_file  # noqa: E402
from melanoma_detection.evaluate import binary_metrics  # noqa: E402
from melanoma_detection.fusion.embeddings import (  # noqa: E402
    embeddings_path,
    load_embeddings,
    merge_embeddings,
)
from melanoma_detection.fusion.evaluate import (  # noqa: E402
    assert_fusion_test_not_yet_evaluated,
    evaluate_fusion_frame,
    fusion_test_paths,
)
from melanoma_detection.fusion.train import (  # noqa: E402
    load_fusion_artifact,
    resolve_fusion_artifact,
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
        f"# Fusion final test — {report['target']}",
        "",
        f"- Generated at: `{report['created_at']}`",
        f"- Model: **{report['model']}**",
        f"- Feature version: `{report['feature_version']}`",
        f"- Embedding model: `{report['embedding_model']}`",
        f"- PCA components: `{report['n_components']}`",
        f"- Artifact: `{report['artifact']}`",
        f"- Split fingerprint: `{report['split_fingerprint']}`",
        f"- F2 threshold (frozen): `{report['thresholds']['f2_threshold']:.6f}`",
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
        "The test split was evaluated once after the model and threshold were",
        "frozen. Do not rerun without an explicit, documented `--force` decision.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=sorted(config.TARGET_SPECS), default="melanoma")
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument("--embeddings", type=Path, default=None)
    parser.add_argument("--interim-path", type=Path, default=None)
    parser.add_argument("--split-path", type=Path, default=None)
    parser.add_argument("--features", choices=sorted(config.FEATURE_SPECS), default="v2")
    parser.add_argument("--metrics-path", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--bootstrap-reps", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing test report.")
    args = parser.parse_args()

    # Check the one-shot guard before loading any test data.
    try:
        assert_fusion_test_not_yet_evaluated(
            args.target,
            force=args.force,
            metrics_path=args.metrics_path,
        )
    except FileExistsError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    metrics_path, markdown_path = fusion_test_paths(
        args.target,
        metrics_path=args.metrics_path,
        markdown_path=args.report_path,
    )

    artifact_path = resolve_fusion_artifact(args.target, args.artifact)
    pipeline, metadata = load_fusion_artifact(artifact_path)

    artifact_target = metadata.get("target")
    if artifact_target != args.target:
        raise RuntimeError(
            f"artifact target is {artifact_target!r}, but --target is {args.target!r}"
        )
    feature_version = metadata.get("feature_version", args.features)
    if feature_version != args.features:
        print(
            f"warning: artifact uses {feature_version!r} features; "
            f"--features is {args.features!r}",
            file=sys.stderr,
        )

    frame = load_modeling_frame(
        args.target,
        interim_path=args.interim_path,
        split_path=args.split_path,
        feature_version=feature_version,
    )
    assert_split_isolation(frame)
    current_fingerprint = split_fingerprint(
        frame, args.target, int(metadata.get("seed", args.seed))
    )
    frozen_fingerprint = metadata.get("split_fingerprint")
    if frozen_fingerprint != current_fingerprint:
        raise RuntimeError(
            "split fingerprint mismatch: the frozen fusion model was trained on a "
            f"different split ({frozen_fingerprint!r} != {current_fingerprint!r})"
        )

    embedding_model = metadata.get("embedding_model", config.FUSION_EMBEDDING_MODEL)
    embeddings_file = args.embeddings or embeddings_path(embedding_model, args.target)
    image_ids, vectors = load_embeddings(embeddings_file)

    validation = frame.loc[frame["split"] == "validation"].reset_index(drop=True)
    test = frame.loc[frame["split"] == "test"].reset_index(drop=True)
    if test.empty:
        raise RuntimeError("test split is empty")
    validation = merge_embeddings(validation, image_ids, vectors)
    test = merge_embeddings(test, image_ids, vectors)

    feature_columns = list(metadata["feature_columns"])
    thresholds = metadata["threshold_selection"]
    threshold = float(thresholds["f2_threshold"])

    validation_evaluation, _ = evaluate_fusion_frame(
        pipeline,
        validation,
        feature_columns,
        threshold,
        bootstrap_reps=args.bootstrap_reps,
        seed=args.seed,
    )
    test_evaluation, test_scores = evaluate_fusion_frame(
        pipeline,
        test,
        feature_columns,
        threshold,
        bootstrap_reps=args.bootstrap_reps,
        seed=args.seed,
    )

    test_at_sensitivity = None
    sensitivity_threshold = thresholds.get("sensitivity_threshold")
    if sensitivity_threshold is not None:
        test_at_sensitivity = {
            "threshold": float(sensitivity_threshold),
            "metrics": binary_metrics(
                test["target"].to_numpy(),
                test_scores,
                threshold=float(sensitivity_threshold),
            ),
        }

    report: dict[str, Any] = {
        "phase": "fusion_final_test",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": args.target,
        "model": metadata.get("model", config.FUSION_MODEL_NAME),
        "feature_version": feature_version,
        "embedding_model": embedding_model,
        "n_components": metadata.get("n_components"),
        "artifact": _relative(artifact_path),
        "artifact_sha256": sha256_file(artifact_path),
        "embeddings": _relative(embeddings_file),
        "embeddings_sha256": sha256_file(embeddings_file),
        "split_fingerprint": frozen_fingerprint,
        "thresholds": thresholds,
        "one_shot": True,
        "validation": validation_evaluation,
        "test": test_evaluation,
        "test_at_sensitivity_threshold": test_at_sensitivity,
        "counts": {
            "validation": {
                "rows": int(len(validation)),
                "lesions": int(validation["lesion_id"].nunique()),
                "positive": int(validation["target"].sum()),
            },
            "test": {
                "rows": int(len(test)),
                "lesions": int(test["lesion_id"].nunique()),
                "positive": int(test["target"].sum()),
            },
        },
        "software": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "command": (
            "uv run --python 3.13 --with pandas --with scikit-learn==1.9.1 "
            "--with numpy --with joblib "
            f"python scripts/evaluate_fusion.py --target {args.target}"
        ),
        "notes": [
            "The test split was evaluated once after freezing the model and threshold.",
            "Confidence intervals resample whole lesions, not individual images.",
            "Not clinical evidence; research use only.",
        ],
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


if __name__ == "__main__":
    raise SystemExit(main())
