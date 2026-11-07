from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config


class DataValidationError(ValueError):
    """Raised when the raw or merged dataset violates the expected contract."""


FEATURE_COLUMNS = [
    "image_id",
    "dx",
    "age",
    "sex",
    "localization",
    "color_1_R",
    "color_1_G",
    "color_1_B",
    "color_1_proportion",
    "color_2_R",
    "color_2_G",
    "color_2_B",
    "color_2_proportion",
    "color_3_R",
    "color_3_G",
    "color_3_B",
    "color_3_proportion",
    "asym_vert_%",
    "asym_horiz_%",
    "Perimeter",
    "Area",
]

METADATA_COLUMNS = [
    "lesion_id",
    "image_id",
    "dx",
    "dx_type",
    "age",
    "sex",
    "localization",
]


def _require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise DataValidationError(f"{name} is missing required columns: {missing}")


def _clean_image_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def load_features(path: str | Path = config.FEATURES_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Feature workbook not found: {path}")

    df = pd.read_excel(path)
    _require_columns(df, FEATURE_COLUMNS, "features")
    df = df.copy()
    df["image_id"] = _clean_image_id(df["image_id"])

    if df["image_id"].isna().any():
        raise DataValidationError("features contains missing image_id values")
    duplicated = df.loc[df["image_id"].duplicated(), "image_id"].tolist()
    if duplicated:
        raise DataValidationError(f"features contains duplicated image_id values: {duplicated[:5]}")

    return df


def load_metadata(path: str | Path = config.METADATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"HAM10000 metadata not found: {path}")

    df = pd.read_csv(path)
    _require_columns(df, METADATA_COLUMNS, "metadata")
    df = df.copy()
    df["image_id"] = _clean_image_id(df["image_id"])

    if df["image_id"].isna().any():
        raise DataValidationError("metadata contains missing image_id values")
    duplicated = df.loc[df["image_id"].duplicated(), "image_id"].tolist()
    if duplicated:
        raise DataValidationError(f"metadata contains duplicated image_id values: {duplicated[:5]}")

    return df


def _values_match(left: pd.Series, right: pd.Series) -> pd.Series:
    left_string = left.astype("string").str.strip()
    right_string = right.astype("string").str.strip()
    return left_string.eq(right_string) | (left_string.isna() & right_string.isna())


def merge_and_validate(features: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Merge lesion_id and original diagnosis, then validate the data contract."""
    features = features.rename(columns={"dx": "dx_current"})
    metadata = metadata.rename(columns={"dx": "dx_original"})

    merged = features.merge(
        metadata[
            [
                "image_id",
                "lesion_id",
                "dx_original",
                "dx_type",
                "age",
                "sex",
                "localization",
            ]
        ],
        on="image_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_meta"),
    )

    unmatched = merged.loc[merged["lesion_id"].isna(), "image_id"].tolist()
    if unmatched:
        raise DataValidationError(
            f"{len(unmatched)} feature images have no metadata match; examples: {unmatched[:5]}"
        )

    # The feature workbook should agree with the official metadata on the
    # shared demographic columns. A mismatch means the two inputs are out of sync.
    for column in ("age", "sex", "localization"):
        meta_column = f"{column}_meta"
        if not _values_match(merged[column], merged[meta_column]).all():
            mismatches = merged.loc[
                ~_values_match(merged[column], merged[meta_column]), ["image_id", column, meta_column]
            ]
            raise DataValidationError(
                f"{column} disagrees between features and metadata for "
                f"{len(mismatches)} rows; examples:\n{mismatches.head()}"
            )
        merged = merged.drop(columns=meta_column)

    if len(merged) != config.EXPECTED_ROWS:
        raise DataValidationError(
            f"expected {config.EXPECTED_ROWS} merged rows, got {len(merged)}"
        )
    if merged["image_id"].duplicated().any():
        raise DataValidationError("merged dataset contains duplicated image_id values")

    lesion_count = merged["lesion_id"].nunique(dropna=True)
    if lesion_count != config.EXPECTED_LESIONS:
        raise DataValidationError(
            f"expected {config.EXPECTED_LESIONS} unique lesions, got {lesion_count}"
        )

    observed_dx = set(merged["dx_original"].dropna().unique())
    expected_dx = set(config.EXPECTED_DX)
    if observed_dx != expected_dx:
        raise DataValidationError(
            f"original dx classes do not match the contract; "
            f"missing={sorted(expected_dx - observed_dx)}, "
            f"unexpected={sorted(observed_dx - expected_dx)}"
        )

    per_lesion_diagnoses = merged.groupby("lesion_id")["dx_original"].nunique()
    inconsistent = per_lesion_diagnoses[per_lesion_diagnoses > 1]
    if len(inconsistent):
        raise DataValidationError(
            f"{len(inconsistent)} lesions have more than one original diagnosis; "
            f"examples: {inconsistent.head().to_dict()}"
        )

    per_lesion_current = merged.groupby("lesion_id")["dx_current"].nunique()
    inconsistent_current = per_lesion_current[per_lesion_current > 1]
    if len(inconsistent_current):
        raise DataValidationError(
            f"{len(inconsistent_current)} lesions have more than one current label; "
            f"examples: {inconsistent_current.head().to_dict()}"
        )

    return merged


def add_target(df: pd.DataFrame, definition: str = config.TARGET_DEFINITION) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add a target column and a human-readable target label.

    The target is the only label column that modelling code should use.
    """
    if definition not in config.TARGET_SPECS:
        raise ValueError(
            f"unknown target definition {definition!r}; "
            f"choose from {sorted(config.TARGET_SPECS)}"
        )

    if "dx_original" not in df.columns:
        raise DataValidationError("add_target requires a 'dx_original' column")

    spec = config.TARGET_SPECS[definition]
    result = df.copy()

    if definition == "multiclass":
        result["target"] = result["dx_original"].astype(str)
        result["target_label"] = result["target"]
        target_info = {
            "definition": definition,
            "description": spec["description"],
            "positive_classes": None,
            "negative_classes": None,
            "labels": {label: label for label in config.EXPECTED_DX},
        }
    else:
        positive = tuple(spec["positive"])
        negative = tuple(spec["negative"])
        if set(positive) & set(negative):
            raise DataValidationError(f"target {definition!r} has overlapping positive/negative classes")
        if set(positive) | set(negative) != set(config.EXPECTED_DX):
            raise DataValidationError(
                f"target {definition!r} does not cover every original dx class"
            )

        result["target"] = result["dx_original"].isin(positive).astype(int)
        result["target_label"] = np.where(
            result["target"] == 1, spec["positive_label"], spec["negative_label"]
        )
        target_info = {
            "definition": definition,
            "description": spec["description"],
            "positive_classes": list(positive),
            "negative_classes": list(negative),
            "labels": {1: spec["positive_label"], 0: spec["negative_label"]},
        }

    if result["target"].isna().any():
        raise DataValidationError("target contains missing values")

    return result, target_info


def load_modeling_frame(
    target: str = config.TARGET_DEFINITION,
    interim_path: str | Path | None = None,
    split_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load features + labels + the Phase 1 split for one target.

    Only ``config.ALL_FEATURES`` are read from the interim dataset. Label and
    audit columns come from the split assignment, so modelling code can
    explicitly select features and cannot accidentally train on a label column.
    """
    interim_path = Path(interim_path or config.INTERIM_DIR / "ham10000_with_lesions.csv")
    split_path = Path(
        split_path or config.PROCESSED_DIR / f"split_assignments_{target}.csv"
    )

    if not interim_path.exists():
        raise FileNotFoundError(
            f"interim dataset not found: {interim_path}; run scripts/build_phase1_dataset.py first"
        )
    if not split_path.exists():
        raise FileNotFoundError(
            f"split assignments not found: {split_path}; run scripts/build_phase1_dataset.py first"
        )

    features = pd.read_csv(interim_path, usecols=["image_id", *config.ALL_FEATURES])
    assignment = pd.read_csv(split_path)

    required_assignment_columns = {
        "image_id",
        "lesion_id",
        "target",
        "target_label",
        "fold",
        "split",
    }
    missing = required_assignment_columns - set(assignment.columns)
    if missing:
        raise DataValidationError(
            f"split assignment is missing columns: {sorted(missing)}"
        )

    if assignment["image_id"].duplicated().any():
        raise DataValidationError("split assignment contains duplicated image_id values")

    frame = features.merge(
        assignment,
        on="image_id",
        how="inner",
        validate="one_to_one",
    )

    if len(frame) != len(assignment):
        raise DataValidationError(
            f"only {len(frame)} of {len(assignment)} split rows matched the interim dataset"
        )

    if frame[list(config.ALL_FEATURES)].columns.duplicated().any():
        raise DataValidationError("feature columns are duplicated after the merge")
    if frame["target"].isna().any():
        raise DataValidationError("target contains missing values")
    if set(frame["split"].unique()) != {"train", "validation", "test"}:
        raise DataValidationError("split column must contain train, validation, and test")

    return frame


def dataset_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Return JSON-friendly dataset counts for the manifest and report."""
    missing = {column: int(df[column].isna().sum()) for column in df.columns if df[column].isna().any()}

    return {
        "rows": int(len(df)),
        "unique_image_ids": int(df["image_id"].nunique()),
        "unique_lesions": int(df["lesion_id"].nunique()),
        "dx_original_counts": {
            str(key): int(value) for key, value in df["dx_original"].value_counts().items()
        },
        "dx_current_counts": {
            str(key): int(value) for key, value in df["dx_current"].value_counts().items()
        },
        "missing_value_counts": missing,
    }


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
