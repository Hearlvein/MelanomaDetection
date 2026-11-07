#!/usr/bin/env python3
"""Fetch the legacy Keras models and report figures from the upstream repository.

The large binaries are intentionally not tracked in git. This script downloads
them on demand and verifies the git blob SHA-1 and byte size.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_ASSETS_DIR = PROJECT_ROOT / "legacy" / "assets"
LEGACY_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "legacy"

UPSTREAM_RAW = "https://raw.githubusercontent.com/Hearlvein/MelanomaDetection/main"

# path -> (url, size, git blob sha1)
ASSETS = {
    "cnn_melanome.keras": (
        f"{UPSTREAM_RAW}/cnn_melanome.keras",
        51_501_171,
        "6b29e07edc98289ec70faa4de049f4a6c5a38dcc",
    ),
    "cnn_vgg16_transfer_melanome.keras": (
        f"{UPSTREAM_RAW}/cnn_vgg16_transfer_melanome.keras",
        60_529_122,
        "210fea1e218ee53acd02a20389070863ce78cb60",
    ),
}

FIGURES = {
    "PCA.png": (
        f"{UPSTREAM_RAW}/images/PCA.png",
        359_988,
        "dcd563c65a2ee3215013ec732c605b55cbfe2ebc",
    ),
    "t-SNE.png": (
        f"{UPSTREAM_RAW}/images/t-SNE.png",
        501_032,
        "70eca9b7f768e0da8afe94c8f0875171f56e7829",
    ),
}


def git_blob_sha1(data: bytes) -> str:
    """Return the git blob SHA-1 for ``data`` (same algorithm as git hash-object)."""
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()


def _download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
        return response.read()


def fetch_one(name: str, url: str, size: int, sha: str, output_dir: Path, *, force: bool) -> bool:
    path = output_dir / name
    if path.exists() and not force:
        data = path.read_bytes()
        if len(data) == size and git_blob_sha1(data) == sha:
            print(f"ok      {path}")
            return True
        print(f"stale   {path} (size or hash mismatch, re-downloading)")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"fetch   {url}")
    data = _download(url)

    if len(data) != size:
        raise RuntimeError(f"size mismatch for {name}: expected {size}, got {len(data)}")
    digest = git_blob_sha1(data)
    if digest != sha:
        raise RuntimeError(f"blob sha mismatch for {name}: expected {sha}, got {digest}")

    path.write_bytes(data)
    print(f"saved   {path} ({size:,} bytes)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", action="store_true", help="Fetch the legacy .keras models.")
    parser.add_argument("--figures", action="store_true", help="Fetch the legacy PCA/t-SNE figures.")
    parser.add_argument("--all", action="store_true", help="Fetch models and figures.")
    parser.add_argument("--force", action="store_true", help="Re-download even if a valid file exists.")
    args = parser.parse_args()

    if not (args.models or args.figures or args.all):
        parser.error("choose --models, --figures, or --all")

    wanted_models = args.models or args.all
    wanted_figures = args.figures or args.all

    try:
        if wanted_models:
            for name, (url, size, sha) in ASSETS.items():
                fetch_one(name, url, size, sha, LEGACY_ASSETS_DIR, force=args.force)
        if wanted_figures:
            for name, (url, size, sha) in FIGURES.items():
                fetch_one(name, url, size, sha, LEGACY_FIGURES_DIR, force=args.force)
    except Exception as error:  # noqa: BLE001 - surface a clear CLI failure
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
