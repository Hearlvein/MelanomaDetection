"""Data loading, validation, target definition, and lesion-grouped splits."""

from __future__ import annotations

import hashlib
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import config

METADATA_COLUMNS = [
    "lesion_id",
    "image_id",
    "dx",
    "dx_type",
    "age",
    "sex",
    "localization",
]
SPLIT_NAMES = ("train", "validation", "test")


class DataValidationError(ValueError):
    """Raised when the raw or merged dataset violates the expected contract."""


class SplitValidationError(ValueError):
    """Raised when a split violates the group or stratification contract."""


def _require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise DataValidationError(f"{name} is missing required columns: {missing}")


def _clean_image_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def load_features(path: str | Path | None = None) -> pd.DataFrame:
    """Load the colour/shape workbook and validate its exact 21-column schema."""
    path = Path(path or config.FEATURES_PATH)
    if not path.exists():
        raise FileNotFoundError(f"feature table not found: {path}")

    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df = df.copy()

    missing = [column for column in config.FEATURE_COLUMNS if column not in df.columns]
    unexpected = [column for column in df.columns if column not in config.FEATURE_COLUMNS]
    if missing or unexpected:
        raise DataValidationError(
            "features do not match the expected schema; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    df = df.loc[:, list(config.FEATURE_COLUMNS)]

    df["image_id"] = _clean_image_id(df["image_id"])
    if df["image_id"].isna().any():
        raise DataValidationError("features contains missing image_id values")
    duplicated = df.loc[df["image_id"].duplicated(), "image_id"].tolist()
    if duplicated:
        raise DataValidationError(
            f"features contains duplicated image_id values: {duplicated[:5]}"
        )

    df.attrs["feature_path"] = str(path)
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
        raise DataValidationError(
            f"metadata contains duplicated image_id values: {duplicated[:5]}"
        )
    return df


def _values_match(left: pd.Series, right: pd.Series) -> pd.Series:
    left_string = left.astype("string").str.strip()
    right_string = right.astype("string").str.strip()
    return left_string.eq(right_string) | (left_string.isna() & right_string.isna())


def merge_and_validate(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    enforce_contract: bool = True,
) -> pd.DataFrame:
    """Merge features with lesion metadata and validate the data contract.

    ``enforce_contract=False`` is intended for small synthetic frames in tests.
    It skips the global row/lesion/class counts but keeps every consistency check.
    """
    features = features.rename(columns={"dx": "dx_current"}).copy()
    metadata = metadata.rename(columns={"dx": "dx_original"}).copy()

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
            f"{len(unmatched)} feature images have no metadata match; "
            f"examples: {unmatched[:5]}"
        )

    # The feature workbook must agree with the official metadata on the shared
    # demographic columns. A mismatch means the two inputs are out of sync.
    for column in ("age", "sex", "localization"):
        meta_column = f"{column}_meta"
        if not _values_match(merged[column], merged[meta_column]).all():
            mismatches = merged.loc[
                ~_values_match(merged[column], merged[meta_column]),
                ["image_id", column, meta_column],
            ]
            raise DataValidationError(
                f"{column} disagrees between features and metadata for "
                f"{len(mismatches)} rows; examples:\n{mismatches.head()}"
            )
        merged = merged.drop(columns=meta_column)

    if enforce_contract and len(merged) != config.EXPECTED_ROWS:
        raise DataValidationError(
            f"expected {config.EXPECTED_ROWS} merged rows, got {len(merged)}"
        )
    if merged["image_id"].duplicated().any():
        raise DataValidationError("merged dataset contains duplicated image_id values")

    lesion_count = merged["lesion_id"].nunique(dropna=True)
    if enforce_contract and lesion_count != config.EXPECTED_LESIONS:
        raise DataValidationError(
            f"expected {config.EXPECTED_LESIONS} unique lesions, got {lesion_count}"
        )

    observed_dx = set(merged["dx_original"].dropna().unique())
    expected_dx = set(config.EXPECTED_DX)
    if enforce_contract and observed_dx != expected_dx:
        raise DataValidationError(
            "original dx classes do not match the contract; "
            f"missing={sorted(expected_dx - observed_dx)}, "
            f"unexpected={sorted(observed_dx - expected_dx)}"
        )

    # A lesion can appear on several images, but its diagnosis must be stable.
    for column in ("dx_original", "dx_current"):
        per_lesion = merged.groupby("lesion_id")[column].nunique()
        inconsistent = per_lesion[per_lesion > 1]
        if len(inconsistent):
            raise DataValidationError(
                f"{len(inconsistent)} lesions have more than one {column} value; "
                f"examples: {inconsistent.head().to_dict()}"
            )

    return merged


def add_target(
    df: pd.DataFrame, definition: str = config.TARGET_DEFINITION
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add the binary melanoma target and a human-readable label."""
    if definition not in config.TARGET_SPECS:
        raise ValueError(
            f"unknown target definition {definition!r}; "
            f"choose from {sorted(config.TARGET_SPECS)}"
        )
    if "dx_original" not in df.columns:
        raise DataValidationError("add_target requires a 'dx_original' column")

    spec = config.TARGET_SPECS[definition]
    positive = tuple(spec["positive"])
    negative = tuple(spec["negative"])
    if set(positive) & set(negative):
        raise DataValidationError(
            f"target {definition!r} has overlapping positive/negative classes"
        )
    if set(positive) | set(negative) != set(config.EXPECTED_DX):
        raise DataValidationError(
            f"target {definition!r} does not cover every original dx class"
        )

    result = df.copy()
    result["target"] = result["dx_original"].isin(positive).astype(int)
    result["target_label"] = np.where(
        result["target"] == 1, spec["positive_label"], spec["negative_label"]
    )

    if result["target"].isna().any():
        raise DataValidationError("target contains missing values")

    target_info = {
        "definition": definition,
        "description": spec["description"],
        "positive_classes": list(positive),
        "negative_classes": list(negative),
        "labels": {1: spec["positive_label"], 0: spec["negative_label"]},
    }
    return result, target_info


def load_split_assignments(
    target: str = config.TARGET_DEFINITION,
    path: str | Path | None = None,
) -> pd.DataFrame:
    """Load the split assignment without any model features."""
    path = Path(path or config.split_assignment_path(target))
    if not path.exists():
        raise FileNotFoundError(
            f"split assignments not found: {path}; run scripts/01_dataset.py first"
        )

    assignment = pd.read_csv(path)
    required = {"image_id", "lesion_id", "target", "target_label", "fold", "split"}
    missing = required - set(assignment.columns)
    if missing:
        raise DataValidationError(
            f"split assignment is missing columns: {sorted(missing)}"
        )
    if assignment["image_id"].duplicated().any():
        raise DataValidationError("split assignment contains duplicated image_id values")
    if assignment["target"].isna().any():
        raise DataValidationError("target contains missing values")
    if set(assignment["split"].unique()) != set(SPLIT_NAMES):
        raise DataValidationError(
            "split column must contain train, validation, and test"
        )
    return assignment


def load_modeling_frame(
    target: str = config.TARGET_DEFINITION,
    interim_path: str | Path | None = None,
    split_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load raw features, labels, and the split assignment for one target."""
    feature_columns = [column for column in config.ALL_FEATURES]
    interim_path = Path(interim_path or config.interim_dataset_path())
    split_path = Path(split_path or config.split_assignment_path(target))

    for path, hint in (
        (interim_path, "scripts/01_dataset.py"),
        (split_path, "scripts/01_dataset.py"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"dataset artifact not found: {path}; run {hint} first")

    features = pd.read_csv(interim_path, usecols=["image_id", *feature_columns])
    assignment = pd.read_csv(split_path)

    required = {"image_id", "lesion_id", "target", "target_label", "fold", "split"}
    missing = required - set(assignment.columns)
    if missing:
        raise DataValidationError(
            f"split assignment is missing columns: {sorted(missing)}"
        )
    if assignment["image_id"].duplicated().any():
        raise DataValidationError("split assignment contains duplicated image_id values")

    frame = features.merge(
        assignment, on="image_id", how="inner", validate="one_to_one"
    )
    if len(frame) != len(assignment):
        raise DataValidationError(
            f"only {len(frame)} of {len(assignment)} split rows matched the interim dataset"
        )
    if frame[list(feature_columns)].columns.duplicated().any():
        raise DataValidationError("feature columns are duplicated after the merge")
    if frame["target"].isna().any():
        raise DataValidationError("target contains missing values")
    if set(frame["split"].unique()) != set(SPLIT_NAMES):
        raise DataValidationError("split column must contain train, validation, and test")

    return frame


def dataset_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Return JSON-friendly dataset counts."""
    missing = {
        column: int(df[column].isna().sum())
        for column in df.columns
        if df[column].isna().any()
    }
    return {
        "rows": int(len(df)),
        "unique_image_ids": int(df["image_id"].nunique()),
        "unique_lesions": int(df["lesion_id"].nunique()),
        "dx_original_counts": {
            str(key): int(value)
            for key, value in df["dx_original"].value_counts().items()
        },
        "dx_current_counts": {
            str(key): int(value)
            for key, value in df["dx_current"].value_counts().items()
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


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def make_grouped_cv(
    n_splits: int = config.N_SPLITS, seed: int = config.SEED
) -> StratifiedGroupKFold:
    """Grouped cross-validation splitter for use inside the training split."""
    if n_splits < 2:
        raise SplitValidationError("n_splits must be at least 2")
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def make_grouped_split(
    df: pd.DataFrame,
    target_col: str = "target",
    group_col: str = "lesion_id",
    n_splits: int = config.N_SPLITS,
    test_fold: int = config.TEST_FOLD,
    val_fold: int = config.VAL_FOLD,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """Create a reproducible 3-way split with lesion-disjoint folds."""
    if target_col not in df.columns:
        raise SplitValidationError(f"missing target column: {target_col!r}")
    if group_col not in df.columns:
        raise SplitValidationError(f"missing group column: {group_col!r}")
    if df[target_col].isna().any():
        raise SplitValidationError("target contains missing values")
    if df[group_col].isna().any():
        raise SplitValidationError("group column contains missing values")
    if n_splits < 3:
        raise SplitValidationError("n_splits must be at least 3")
    if not 0 <= test_fold < n_splits:
        raise SplitValidationError(f"test_fold must be in [0, {n_splits - 1}]")
    if not 0 <= val_fold < n_splits:
        raise SplitValidationError(f"val_fold must be in [0, {n_splits - 1}]")
    if test_fold == val_fold:
        raise SplitValidationError("test_fold and val_fold must be different")

    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    fold_assignment = np.full(len(df), -1, dtype=int)
    for fold, (_, held_out_index) in enumerate(
        splitter.split(df, y=df[target_col], groups=df[group_col])
    ):
        fold_assignment[held_out_index] = fold

    if (fold_assignment < 0).any():
        raise SplitValidationError("some rows were not assigned to a fold")

    split_assignment = np.select(
        [fold_assignment == test_fold, fold_assignment == val_fold],
        ["test", "validation"],
        default="train",
    )
    result = df.copy()
    result["fold"] = fold_assignment
    result["split"] = pd.Series(split_assignment, index=result.index)
    return result


def validate_split(
    df: pd.DataFrame,
    target_col: str = "target",
    group_col: str = "lesion_id",
    max_prevalence_diff: float = 0.03,
) -> dict[str, Any]:
    """Validate group isolation, coverage, and approximate stratification."""
    required = {"image_id", group_col, target_col, "split", "fold"}
    missing_columns = required - set(df.columns)
    if missing_columns:
        raise SplitValidationError(
            f"missing split columns: {sorted(missing_columns)}"
        )

    observed_splits = set(df["split"].unique())
    if observed_splits != set(SPLIT_NAMES):
        raise SplitValidationError(
            f"expected split values {sorted(SPLIT_NAMES)}, got {sorted(observed_splits)}"
        )
    if df["image_id"].duplicated().any():
        raise SplitValidationError("image_id is duplicated across the split assignment")

    group_sets = {
        split: set(df.loc[df["split"] == split, group_col].unique())
        for split in SPLIT_NAMES
    }
    overlap: dict[str, int] = {}
    for left, right in combinations(SPLIT_NAMES, 2):
        shared = group_sets[left] & group_sets[right]
        overlap[f"{left}__{right}"] = len(shared)
        if shared:
            raise SplitValidationError(
                f"{len(shared)} lesions overlap between {left} and {right}: "
                f"{sorted(shared)[:5]}"
            )

    overall_prevalence = df[target_col].value_counts(normalize=True, dropna=False)
    prevalence: dict[str, dict[str, float]] = {}
    max_diff = 0.0

    for split in SPLIT_NAMES:
        subset = df.loc[df["split"] == split]
        split_prevalence = subset[target_col].value_counts(normalize=True, dropna=False)
        split_prevalence = split_prevalence.reindex(
            overall_prevalence.index, fill_value=0.0
        )
        diff = float((split_prevalence - overall_prevalence).abs().max())
        max_diff = max(max_diff, diff)
        prevalence[split] = {
            str(label): float(value) for label, value in split_prevalence.items()
        }
        missing_labels = set(overall_prevalence.index) - set(
            subset[target_col].unique()
        )
        if missing_labels:
            raise SplitValidationError(
                f"{split} is missing target labels: {sorted(missing_labels)}"
            )

    if max_diff > max_prevalence_diff:
        raise SplitValidationError(
            f"target prevalence differs by up to {max_diff:.4f}, "
            f"above the allowed {max_prevalence_diff:.4f}"
        )

    counts = {
        split: {
            "rows": int((df["split"] == split).sum()),
            "lesions": int(len(group_sets[split])),
            "target_counts": {
                str(label): int(value)
                for label, value in df.loc[df["split"] == split, target_col]
                .value_counts()
                .items()
            },
        }
        for split in SPLIT_NAMES
    }

    return {
        "counts": counts,
        "group_overlap": overlap,
        "prevalence": prevalence,
        "max_prevalence_diff": max_diff,
        "prevalence_tolerance": max_prevalence_diff,
    }


def split_fingerprint(
    df: pd.DataFrame, target_definition: str, seed: int
) -> str:
    """Stable SHA-256 fingerprint of the image/fold/split assignment."""
    digest = hashlib.sha256()
    digest.update(f"target={target_definition}|seed={seed}\n".encode("utf-8"))
    ordered = df.sort_values("image_id")[["image_id", "split", "fold"]]
    for row in ordered.itertuples(index=False):
        digest.update(
            f"{row.image_id}|{row.split}|{int(row.fold)}\n".encode("utf-8")
        )
    return digest.hexdigest()
