#!/usr/bin/env python3
"""Check that the external HAM10000 images and Tschandl masks are present.

This is a read-only check. It does not download anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTERNAL = PROJECT_ROOT / "data" / "external" / "HAM10000_Dataset"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "raw" / "HAM10000_metadata.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-dir", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--examples", type=int, default=5)
    args = parser.parse_args()

    if not args.metadata.exists():
        print(f"error: metadata not found: {args.metadata}", file=sys.stderr)
        return 1

    metadata = pd.read_csv(args.metadata)
    image_dir = args.external_dir / "HAM10000"
    mask_dir = args.external_dir / "HAM10000_segmentations_lesion_tschandl"

    print(f"metadata: {args.metadata} ({len(metadata):,} rows)")
    print(f"images:   {image_dir}")
    print(f"masks:    {mask_dir}")

    missing_images = [
        image_id for image_id in metadata["image_id"] if not (image_dir / f"{image_id}.jpg").exists()
    ]
    missing_masks = [
        image_id
        for image_id in metadata["image_id"]
        if not (mask_dir / f"{image_id}_segmentation.png").exists()
    ]

    print()
    print(f"missing images: {len(missing_images):,} / {len(metadata):,}")
    print(f"missing masks:  {len(missing_masks):,} / {len(metadata):,}")

    for label, missing in (("image", missing_images), ("mask", missing_masks)):
        if missing:
            examples = ", ".join(missing[: args.examples])
            print(f"  first missing {label} ids: {examples}")

    if missing_images or missing_masks:
        print()
        print("External data is incomplete. See data/README.md for the expected layout.")
        return 1

    print()
    print("External data is complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
