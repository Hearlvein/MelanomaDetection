from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from melanoma_detection.schema import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS_V2,
    NUMERIC_FEATURES_V2,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTRACT_SCRIPT = PROJECT_ROOT / "scripts" / "extract_embeddings.py"
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_fusion.py"
EVALUATE_SCRIPT = PROJECT_ROOT / "scripts" / "evaluate_fusion.py"


def _make_dataset(root: Path) -> tuple[Path, Path, Path]:
    image_dir = root / "HAM10000"
    image_dir.mkdir(parents=True)
    rng = np.random.default_rng(11)
    splits = ["train"] * 10 + ["validation"] * 3 + ["test"] * 3
    assignment_rows = []
    feature_rows = []
    for index, split in enumerate(splits):
        image_id = f"ISIC_{index:04d}"
        lesion_id = f"HAM_{index:07d}"
        target = index % 2
        image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        Image.fromarray(image, mode="RGB").save(image_dir / f"{image_id}.jpg")

        assignment_rows.append(
            {
                "image_id": image_id,
                "lesion_id": lesion_id,
                "target": target,
                "target_label": "melanoma" if target else "non_melanoma",
                "fold": index % 5,
                "split": split,
            }
        )

        feature_row: dict[str, object] = {
            "image_id": image_id,
            "lesion_id": lesion_id,
            "dx": "mel" if target else "nv",
        }
        for column in NUMERIC_FEATURES_V2:
            feature_row[column] = float(rng.normal() + target)
        for column in CATEGORICAL_FEATURES:
            feature_row[column] = "male" if column == "sex" else "back"
        feature_rows.append(feature_row)

    split_path = root / "split.csv"
    pd.DataFrame(assignment_rows).to_csv(split_path, index=False)

    interim_path = root / "features_v2.csv"
    pd.DataFrame(feature_rows, columns=list(FEATURE_COLUMNS_V2)).to_csv(
        interim_path, index=False
    )
    return image_dir, split_path, interim_path


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        timeout=600,
    )


def test_fusion_cli_extract_train_evaluate_and_one_shot_guard(tmp_path) -> None:
    pytest.importorskip("tensorflow")

    image_dir, split_path, interim_path = _make_dataset(tmp_path)
    embeddings_path = tmp_path / "embeddings.npz"
    manifest_path = tmp_path / "embeddings.json"

    extract_result = _run(
        str(EXTRACT_SCRIPT),
        "--target", "melanoma",
        "--model", "vgg16",
        "--weights", "none",
        "--features", "v2",
        "--split-path", str(split_path),
        "--image-dir", str(image_dir),
        "--image-size", "32", "32",
        "--batch-size", "4",
        "--output", str(embeddings_path),
        "--manifest", str(manifest_path),
    )
    assert extract_result.returncode == 0, extract_result.stderr
    assert embeddings_path.exists()
    assert manifest_path.exists()

    artifact = tmp_path / "fusion.joblib"
    training_report = tmp_path / "train.json"
    train_result = _run(
        str(TRAIN_SCRIPT),
        "--target", "melanoma",
        "--features", "v2",
        "--model", "vgg16",
        "--interim-path", str(interim_path),
        "--split-path", str(split_path),
        "--embeddings", str(embeddings_path),
        "--n-components", "2",
        "--n-splits", "2",
        "--bootstrap-reps", "5",
        "--output", str(artifact),
        "--report", str(training_report),
        "--no-pointer",
    )
    assert train_result.returncode == 0, train_result.stderr
    assert artifact.exists()
    assert artifact.with_suffix(".json").exists()
    assert training_report.exists()
    assert training_report.with_suffix(".md").exists()

    metrics_path = tmp_path / "test.json"
    report_path = tmp_path / "test.md"
    evaluate_arguments = [
        str(EVALUATE_SCRIPT),
        "--target", "melanoma",
        "--features", "v2",
        "--artifact", str(artifact),
        "--embeddings", str(embeddings_path),
        "--interim-path", str(interim_path),
        "--split-path", str(split_path),
        "--bootstrap-reps", "5",
        "--metrics-path", str(metrics_path),
        "--report-path", str(report_path),
    ]
    evaluate_result = _run(*evaluate_arguments)
    assert evaluate_result.returncode == 0, evaluate_result.stderr
    assert metrics_path.exists()
    assert report_path.exists()
    assert "Fusion final test" in report_path.read_text(encoding="utf-8")

    second_result = _run(*evaluate_arguments)
    assert second_result.returncode == 1
    assert "already exists" in second_result.stderr
