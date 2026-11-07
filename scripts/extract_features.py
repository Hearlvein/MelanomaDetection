#!/usr/bin/env python3
"""Extract V2 handcrafted features from HAM10000 images and lesion masks.

This is the supported replacement for the extraction cells in
``legacy/notebooks/FeaturesExtraction.ipynb``. It keeps ``lesion_id`` so the
output can feed the lesion-grouped split and never leaks a lesion across folds.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from melanoma_detection.extraction import (  # noqa: E402
    FEATURE_COLUMNS_V2,
    extract_features_for_frame,
)
from melanoma_detection.extraction.extract import (  # noqa: E402
    IMAGE_DIR_NAME,
    MASK_DIR_NAME,
    write_manifest,
)

DEFAULT_EXTERNAL = PROJECT_ROOT / "data" / "external" / "HAM10000_Dataset"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "raw" / "HAM10000_metadata.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "interim" / "features_v2.csv"
DEFAULT_MANIFEST = PROJECT_ROOT / "reports" / "metrics" / "features_v2_manifest.json"


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--external-dir", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--n-colors", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N rows.")
    parser.add_argument("--no-progress", action="store_true", help="Disable the progress bar.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output.")
    args = parser.parse_args()

    if not args.metadata.exists():
        print(f"error: metadata not found: {args.metadata}", file=sys.stderr)
        return 1

    image_dir = args.external_dir / IMAGE_DIR_NAME
    mask_dir = args.external_dir / MASK_DIR_NAME
    for label, directory in (("images", image_dir), ("masks", mask_dir)):
        if not directory.exists():
            print(f"error: {label} directory not found: {directory}", file=sys.stderr)
            print("See data/README.md, then run scripts/check_external_data.py.", file=sys.stderr)
            return 1

    if args.output.exists() and not args.overwrite:
        print(
            f"error: output already exists: {args.output}. "
            "Pass --overwrite to regenerate it.",
            file=sys.stderr,
        )
        return 1

    print(f"Loading metadata: {args.metadata}")
    metadata = pd.read_csv(args.metadata)

    print(
        f"Extracting features for {len(metadata) if args.limit is None else min(len(metadata), args.limit)} images..."
    )
    features = extract_features_for_frame(
        metadata,
        args.external_dir,
        n_colors=args.n_colors,
        limit=args.limit,
        progress=not args.no_progress,
    )

    missing_columns = [column for column in FEATURE_COLUMNS_V2 if column not in features.columns]
    if missing_columns:
        raise RuntimeError(f"extraction output is missing columns: {missing_columns}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False)

    command = (
        "uv run --python 3.13 --with pandas --with scikit-learn==1.9.1 "
        "--with scikit-image --with pillow --with tqdm "
        f"python scripts/extract_features.py --n-colors {args.n_colors}"
    )
    manifest = write_manifest(
        args.output,
        metadata_path=args.metadata,
        schema="v2",
        rows=len(features),
        lesions=int(features["lesion_id"].nunique(dropna=True)),
        n_colors=args.n_colors,
        command=command,
    )
    manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    manifest["external_dir"] = _relative(args.external_dir)
    manifest["output"]["path"] = _relative(args.output)

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote features: {args.output} ({len(features):,} rows)")
    print(f"Wrote manifest: {args.manifest}")
    print(f"Unique lesions: {manifest['lesions']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
