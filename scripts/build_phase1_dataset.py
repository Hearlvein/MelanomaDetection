#!/usr/bin/env python3
"""Phase 1: merge HAM10000 metadata, define the target, and create a lesion-grouped split."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from melanoma_detection import config  # noqa: E402
from melanoma_detection.data import (  # noqa: E402
    add_target,
    dataset_summary,
    load_features,
    load_metadata,
    merge_and_validate,
    sha256_file,
)
from melanoma_detection.splits import (  # noqa: E402
    SPLIT_NAMES,
    make_grouped_split,
    split_fingerprint,
    validate_split,
)


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


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _class_order(target_info: dict[str, object]) -> list[str]:
    if target_info["definition"] == "multiclass":
        return list(config.EXPECTED_DX)
    labels = target_info["labels"]
    assert isinstance(labels, dict)
    return [str(labels[0]), str(labels[1])]


def render_summary(
    *,
    target_info: dict[str, object],
    split_info: dict[str, object],
    dataset_info: dict[str, object],
    manifest: dict[str, object],
    assignment: pd.DataFrame,
    command: str,
) -> str:
    split_counts = split_info["counts"]
    assert isinstance(split_counts, dict)

    positive_classes = target_info["positive_classes"]
    negative_classes = target_info["negative_classes"]

    split_rows: list[list[object]] = []
    for split in SPLIT_NAMES:
        counts = split_counts[split]
        split_rows.append(
            [
                split,
                counts["rows"],
                counts["lesions"],
                f"{counts['rows'] / len(assignment):.3%}",
            ]
        )

    target_counts = (
        pd.crosstab(assignment["split"], assignment["target_label"])
        .reindex(index=list(SPLIT_NAMES), columns=_class_order(target_info), fill_value=0)
        .reset_index()
    )
    target_rows = target_counts.values.tolist()

    dx_order = list(config.EXPECTED_DX)
    dx_counts = (
        pd.crosstab(assignment["split"], assignment["dx_original"])
        .reindex(index=list(SPLIT_NAMES), columns=dx_order, fill_value=0)
        .reset_index()
    )
    dx_rows = dx_counts.values.tolist()

    overlap = split_info["group_overlap"]
    assert isinstance(overlap, dict)

    lines = [
        "# Phase 1 — Data integrity and grouped split",
        "",
        f"- Generated at: `{manifest['created_at']}`",
        f"- Target definition: **{target_info['definition']}**",
        f"- Description: {target_info['description']}",
        f"- Positive classes: {positive_classes}",
        f"- Negative classes: {negative_classes}",
        f"- Random seed: `{manifest['split']['seed']}`",
        f"- Split fingerprint: `{manifest['split']['fingerprint']}`",
        "",
        "## Inputs",
        "",
        _markdown_table(
            ["File", "SHA-256"],
            [[name, digest] for name, digest in manifest["inputs"].items()],
        ),
        "",
        "## Dataset checks",
        "",
        _markdown_table(
            ["Check", "Result"],
            [
                ["Rows", dataset_info["rows"]],
                ["Unique image IDs", dataset_info["unique_image_ids"]],
                ["Unique lesions", dataset_info["unique_lesions"]],
                ["Original diagnosis classes", ", ".join(config.EXPECTED_DX)],
                ["Missing values", dataset_info["missing_value_counts"] or "none"],
            ],
        ),
        "",
        "## Split summary",
        "",
        _markdown_table(["Split", "Rows", "Lesions", "Share"], split_rows),
        "",
        "### Target distribution by split",
        "",
        _markdown_table(["Split", *_class_order(target_info)], target_rows),
        "",
        "### Original diagnosis distribution by split",
        "",
        _markdown_table(["Split", *dx_order], dx_rows),
        "",
        "## Leakage checks",
        "",
        _markdown_table(
            ["Check", "Result"],
            [
                ["Lesion overlap train/validation", overlap["train__validation"]],
                ["Lesion overlap train/test", overlap["train__test"]],
                ["Lesion overlap validation/test", overlap["validation__test"]],
                ["Max target prevalence difference", f"{split_info['max_prevalence_diff']:.4f}"],
                ["Prevalence tolerance", f"{split_info['prevalence_tolerance']:.4f}"],
            ],
        ),
        "",
        "## Outputs",
        "",
        _markdown_table(
            ["Artifact", "Path"],
            [[name, path] for name, path in manifest["outputs"].items()],
        ),
        "",
        "## Notes for later phases",
        "",
        "- Use `split` for the final holdout.",
        "- Run grouped cross-validation **inside `split == 'train'`** (recompute `StratifiedGroupKFold` on the training rows, or use the precomputed training folds 2, 3, and 4 as a 3-fold grouped CV). Never include validation or test rows in CV.",
        "- `lesion_id` is a grouping key only. It must never be a model feature.",
        "- `dx_original`, `dx_current`, `dx_type`, `target`, `target_label`, `fold`, and `split` are label/audit columns. They must never be model features.",
        "- The Phase 1 default target is `melanoma` (melanoma vs all non-melanoma), matching the current notebooks. Use `--target malignant` or `--target multiclass` to regenerate the split for another task.",
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
    parser.add_argument(
        "--target",
        choices=sorted(config.TARGET_SPECS),
        default=config.TARGET_DEFINITION,
        help="Target definition used for stratification and the split artifact.",
    )
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--n-splits", type=int, default=config.N_SPLITS)
    parser.add_argument("--test-fold", type=int, default=config.TEST_FOLD)
    parser.add_argument("--val-fold", type=int, default=config.VAL_FOLD)
    parser.add_argument(
        "--prevalence-tolerance",
        type=float,
        default=0.03,
        help="Maximum allowed absolute target-prevalence difference across splits.",
    )
    args = parser.parse_args()

    config.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading feature workbook and official metadata...")
    features = load_features()
    metadata = load_metadata()

    print("Merging metadata and validating the data contract...")
    merged = merge_and_validate(features, metadata)
    merged, target_info = add_target(merged, args.target)

    print("Building lesion-grouped split...")
    assignment = make_grouped_split(
        merged,
        target_col="target",
        group_col="lesion_id",
        n_splits=args.n_splits,
        test_fold=args.test_fold,
        val_fold=args.val_fold,
        seed=args.seed,
    )
    split_info = validate_split(
        assignment,
        target_col="target",
        group_col="lesion_id",
        max_prevalence_diff=args.prevalence_tolerance,
    )

    interim_path = config.INTERIM_DIR / "ham10000_with_lesions.csv"
    assignment_path = config.PROCESSED_DIR / f"split_assignments_{args.target}.csv"
    manifest_path = config.METRICS_DIR / f"phase1_manifest_{args.target}.json"
    summary_path = config.METRICS_DIR / f"phase1_summary_{args.target}.md"
    active_target_path = config.PROCESSED_DIR / "active_target.txt"

    merged.to_csv(interim_path, index=False)

    assignment_columns = [
        "image_id",
        "lesion_id",
        "dx_original",
        "dx_current",
        "dx_type",
        "target",
        "target_label",
        "fold",
        "split",
    ]
    assignment[assignment_columns].to_csv(assignment_path, index=False)

    fingerprint = split_fingerprint(assignment, args.target, args.seed)
    fractions = {
        split: float((assignment["split"] == split).mean()) for split in SPLIT_NAMES
    }

    manifest = {
        "phase": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": target_info,
        "split": {
            "method": "StratifiedGroupKFold",
            "n_splits": args.n_splits,
            "test_fold": args.test_fold,
            "val_fold": args.val_fold,
            "seed": args.seed,
            "fractions": fractions,
            "fingerprint": fingerprint,
        },
        "dataset": dataset_summary(merged),
        "validation": split_info,
        "inputs": {
            _relative(config.FEATURES_PATH): sha256_file(config.FEATURES_PATH),
            _relative(config.METADATA_PATH): sha256_file(config.METADATA_PATH),
        },
        "outputs": {
            "interim_dataset": _relative(interim_path),
            "split_assignments": _relative(assignment_path),
            "manifest": _relative(manifest_path),
            "summary": _relative(summary_path),
        },
        "software": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
        },
    }

    command = (
        "uv run --python 3.13 "
        "--with pandas --with openpyxl --with scikit-learn==1.9.1 "
        f"python scripts/build_phase1_dataset.py --target {args.target} "
        f"--seed {args.seed} --n-splits {args.n_splits} "
        f"--test-fold {args.test_fold} --val-fold {args.val_fold}"
    )

    summary = render_summary(
        target_info=target_info,
        split_info=split_info,
        dataset_info=manifest["dataset"],
        manifest=manifest,
        assignment=assignment,
        command=command,
    )
    summary_path.write_text(summary, encoding="utf-8")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    active_target_path.write_text(
        "\n".join(
            [
                f"target={args.target}",
                f"split_file={_relative(assignment_path)}",
                f"fingerprint={fingerprint}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Wrote interim dataset: {interim_path}")
    print(f"Wrote split assignments: {assignment_path}")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote summary: {summary_path}")
    print()
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
