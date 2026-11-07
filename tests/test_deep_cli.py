from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_deep.py"
EVALUATE_SCRIPT = PROJECT_ROOT / "scripts" / "evaluate_deep.py"


def _make_dataset(root: Path) -> tuple[Path, Path]:
    image_dir = root / "HAM10000"
    image_dir.mkdir(parents=True)
    rng = np.random.default_rng(3)
    splits = ["train"] * 4 + ["validation"] * 2 + ["test"] * 2
    rows = []
    for index, split in enumerate(splits):
        image_id = f"ISIC_{index:04d}"
        image = rng.integers(0, 256, size=(24, 24, 3), dtype=np.uint8)
        Image.fromarray(image, mode="RGB").save(image_dir / f"{image_id}.jpg")
        target = index % 2
        rows.append(
            {
                "image_id": image_id,
                "lesion_id": f"HAM_{index:07d}",
                "target": target,
                "target_label": "melanoma" if target else "non_melanoma",
                "fold": index % 3,
                "split": split,
            }
        )
    split_path = root / "split.csv"
    pd.DataFrame(rows).to_csv(split_path, index=False)
    return image_dir, split_path


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        timeout=300,
    )


def test_deep_cli_train_evaluate_and_one_shot_guard(tmp_path) -> None:
    pytest.importorskip("tensorflow")

    image_dir, split_path = _make_dataset(tmp_path)
    artifact = tmp_path / "model.keras"
    training_report = tmp_path / "train.json"

    train_result = _run(
        str(TRAIN_SCRIPT),
        "--target", "melanoma",
        "--model", "custom_cnn",
        "--split-path", str(split_path),
        "--image-dir", str(image_dir),
        "--image-size", "24", "24",
        "--batch-size", "2",
        "--epochs", "1",
        "--weights", "none",
        "--class-weight", "none",
        "--no-augment",
        "--no-pointer",
        "--output", str(artifact),
        "--report", str(training_report),
    )
    assert train_result.returncode == 0, train_result.stderr
    assert artifact.exists()
    assert artifact.with_suffix(".json").exists()
    assert training_report.exists()

    metrics_path = tmp_path / "test.json"
    report_path = tmp_path / "test.md"
    evaluate_arguments = [
        str(EVALUATE_SCRIPT),
        "--target", "melanoma",
        "--model", "custom_cnn",
        "--artifact", str(artifact),
        "--split-path", str(split_path),
        "--image-dir", str(image_dir),
        "--image-size", "24", "24",
        "--bootstrap-reps", "10",
        "--metrics-path", str(metrics_path),
        "--report-path", str(report_path),
    ]
    evaluate_result = _run(*evaluate_arguments)
    assert evaluate_result.returncode == 0, evaluate_result.stderr
    assert metrics_path.exists()
    assert report_path.exists()
    assert "Deep final test" in report_path.read_text(encoding="utf-8")

    second_result = _run(*evaluate_arguments)
    assert second_result.returncode == 1
    assert "already exists" in second_result.stderr
