from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from melanoma_detection.splits import (
    SplitValidationError,
    make_grouped_cv,
    make_grouped_split,
    split_fingerprint,
    validate_split,
)


def _toy_data(n_groups: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for group in range(n_groups):
        target = int(rng.random() < 0.2)
        image_count = int(rng.integers(1, 4))
        for image_index in range(image_count):
            rows.append(
                {
                    "image_id": f"img_{group:04d}_{image_index}",
                    "lesion_id": f"les_{group:04d}",
                    "target": target,
                }
            )
    return pd.DataFrame(rows)


def test_split_is_group_disjoint_and_complete() -> None:
    data = _toy_data()
    assigned = make_grouped_split(data, seed=42)
    summary = validate_split(assigned)

    assert set(assigned["split"]) == {"train", "validation", "test"}
    assert assigned["fold"].between(0, 4).all()
    assert len(assigned) == len(data)
    assert summary["group_overlap"] == {
        "train__validation": 0,
        "train__test": 0,
        "validation__test": 0,
    }
    assert summary["max_prevalence_diff"] <= summary["prevalence_tolerance"]


def test_split_is_deterministic_for_the_same_seed() -> None:
    data = _toy_data()

    first = make_grouped_split(data, seed=42)
    second = make_grouped_split(data, seed=42)
    pd.testing.assert_frame_equal(first, second)

    different = make_grouped_split(data, seed=7)
    assert not first["split"].equals(different["split"])


def test_split_keeps_lesions_together() -> None:
    data = _toy_data()
    assigned = make_grouped_split(data, seed=42)

    lesions_per_split = assigned.groupby("lesion_id")["split"].nunique()
    assert (lesions_per_split == 1).all()


def test_grouped_cv_keeps_lesions_disjoint() -> None:
    data = _toy_data()
    splitter = make_grouped_cv(n_splits=5, seed=42)

    for train_index, validation_index in splitter.split(
        data, y=data["target"], groups=data["lesion_id"]
    ):
        train_lesions = set(data.iloc[train_index]["lesion_id"])
        validation_lesions = set(data.iloc[validation_index]["lesion_id"])
        assert train_lesions.isdisjoint(validation_lesions)


def test_invalid_fold_configuration_is_rejected() -> None:
    data = _toy_data()
    with pytest.raises(SplitValidationError):
        make_grouped_split(data, test_fold=1, val_fold=1)

    with pytest.raises(SplitValidationError):
        make_grouped_split(data, test_fold=5, val_fold=1)

    with pytest.raises(SplitValidationError):
        make_grouped_split(data, n_splits=2)


def test_fingerprint_is_stable_and_target_specific() -> None:
    data = _toy_data()
    assigned = make_grouped_split(data, seed=42)

    first = split_fingerprint(assigned, "melanoma", 42)
    second = split_fingerprint(assigned, "melanoma", 42)
    other_target = split_fingerprint(assigned, "malignant", 42)

    assert first == second
    assert first != other_target
