#!/usr/bin/env python3
"""Step 1: merge HAM10000 metadata and build the lesion-grouped split."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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

    print("Loading the feature workbook and official metadata...")
    features = load_features()
    metadata = load_metadata()

    print("Merging and validating the data contract...")
    merged = merge_and_validate(features, metadata)
    merged, target_info = add_target(merged, config.TARGET_DEFINITION)

    print("Building the lesion-grouped split...")
    assignment = make_grouped_split(
        merged,
        target_col="target",
        group_col=config.GROUP_COLUMN,
        n_splits=args.n_splits,
        test_fold=args.test_fold,
        val_fold=args.val_fold,
        seed=args.seed,
    )
    split_info = validate_split(
        assignment,
        target_col="target",
        group_col=config.GROUP_COLUMN,
        max_prevalence_diff=args.prevalence_tolerance,
    )

    interim_path = config.interim_dataset_path()
    assignment_path = config.split_assignment_path()
    report_path = config.dataset_report_path()

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

    fingerprint = split_fingerprint(
        assignment, config.TARGET_DEFINITION, args.seed
    )
    fractions = {
        split: float((assignment["split"] == split).mean())
        for split in ("train", "validation", "test")
    }
    command = (
        "uv run --python 3.13 --with pandas --with openpyxl "
        "--with scikit-learn==1.9.1 python scripts/01_dataset.py "
        f"--seed {args.seed} --n-splits {args.n_splits} "
        f"--test-fold {args.test_fold} --val-fold {args.val_fold}"
    )

    report = {
        "stage": "dataset",
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
            "report": _relative(report_path),
        },
        "software": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "command": command,
    }

    report_path.write_text(
        json.dumps(report, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote interim dataset: {interim_path}")
    print(f"Wrote split assignments: {assignment_path}")
    print(f"Wrote dataset report: {report_path}")
    print(f"Split fingerprint: {fingerprint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
