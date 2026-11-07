from __future__ import annotations

import hashlib
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import config

SPLIT_NAMES = ("train", "validation", "test")


class SplitValidationError(ValueError):
    """Raised when a split violates the group or stratification contract."""


def make_grouped_cv(
    n_splits: int = config.N_SPLITS,
    seed: int = config.SEED,
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
    """Create a reproducible 3-way split with lesion-disjoint folds.

    ``StratifiedGroupKFold`` produces K disjoint folds. We reserve one fold for
    test and one for validation; the remaining folds form the training set.
    With the default K=5 this gives an approximate 60/20/20 split.
    """
    if target_col not in df.columns:
        raise SplitValidationError(f"missing target column: {target_col!r}")
    if group_col not in df.columns:
        raise SplitValidationError(f"missing group column: {group_col!r}")
    if df[target_col].isna().any():
        raise SplitValidationError("target contains missing values")
    if df[group_col].isna().any():
        raise SplitValidationError("group column contains missing values")
    if n_splits < 3:
        raise SplitValidationError("n_splits must be at least 3 to create train/validation/test")
    if not 0 <= test_fold < n_splits:
        raise SplitValidationError(f"test_fold must be in [0, {n_splits - 1}]")
    if not 0 <= val_fold < n_splits:
        raise SplitValidationError(f"val_fold must be in [0, {n_splits - 1}]")
    if test_fold == val_fold:
        raise SplitValidationError("test_fold and val_fold must be different")

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
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
        raise SplitValidationError(f"missing split columns: {sorted(missing_columns)}")

    observed_splits = set(df["split"].unique())
    if observed_splits != set(SPLIT_NAMES):
        raise SplitValidationError(
            f"expected split values {sorted(SPLIT_NAMES)}, got {sorted(observed_splits)}"
        )

    if df["image_id"].duplicated().any():
        raise SplitValidationError("image_id is duplicated across the split assignment")

    group_sets = {
        split: set(df.loc[df["split"] == split, group_col].unique()) for split in SPLIT_NAMES
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
        split_prevalence = split_prevalence.reindex(overall_prevalence.index, fill_value=0.0)
        diff = float((split_prevalence - overall_prevalence).abs().max())
        max_diff = max(max_diff, diff)
        prevalence[split] = {
            str(label): float(value) for label, value in split_prevalence.items()
        }

        missing_labels = set(overall_prevalence.index) - set(subset[target_col].unique())
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
    df: pd.DataFrame,
    target_definition: str,
    seed: int,
) -> str:
    """Stable SHA-256 fingerprint of the image/fold/split assignment."""
    digest = hashlib.sha256()
    digest.update(f"target={target_definition}|seed={seed}\n".encode("utf-8"))
    ordered = df.sort_values("image_id")[["image_id", "split", "fold"]]
    for row in ordered.itertuples(index=False):
        digest.update(f"{row.image_id}|{row.split}|{int(row.fold)}\n".encode("utf-8"))
    return digest.hexdigest()
