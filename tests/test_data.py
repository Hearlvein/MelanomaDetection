from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from melanoma_detection import config
from melanoma_detection.data import (
    DataValidationError,
    SplitValidationError,
    add_target,
    load_features,
    load_metadata,
    load_modeling_frame,
    make_grouped_split,
    merge_and_validate,
    split_fingerprint,
    validate_split,
)


def _toy_features(image_ids: list[str]) -> pd.DataFrame:
    data: dict[str, object] = {"image_id": image_ids, "dx": ["mel"] * len(image_ids)}
    for offset, column in enumerate(config.NUMERIC_FEATURES):
        data[column] = np.arange(len(image_ids), dtype=float) + offset
    data["sex"] = ["male"] * len(image_ids)
    data["localization"] = ["back"] * len(image_ids)
    return pd.DataFrame(data)[list(config.FEATURE_COLUMNS)]


def _toy_metadata(image_ids: list[str]) -> pd.DataFrame:
    features = _toy_features(image_ids)
    return pd.DataFrame(
        {
            "lesion_id": [f"les_{index // 2}" for index in range(len(image_ids))],
            "image_id": image_ids,
            "dx": ["mel"] * len(image_ids),
            "dx_type": ["histo"] * len(image_ids),
            "age": features["age"].to_numpy(),
            "sex": features["sex"].to_numpy(),
            "localization": features["localization"].to_numpy(),
        }
    )


def _toy_data(n_groups: int = 200, seed: int = 42) -> pd.DataFrame:
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


def test_load_features_accepts_the_exact_schema(tmp_path) -> None:
    path = tmp_path / "features.csv"
    _toy_features(["img_1", "img_2"]).to_csv(path, index=False)

    loaded = load_features(path)

    assert list(loaded.columns) == list(config.FEATURE_COLUMNS)
    assert loaded["image_id"].tolist() == ["img_1", "img_2"]


def test_load_features_rejects_an_unexpected_column(tmp_path) -> None:
    path = tmp_path / "features.csv"
    frame = _toy_features(["img_1"])
    frame["leak"] = 1
    frame.to_csv(path, index=False)

    with pytest.raises(DataValidationError, match="unexpected"):
        load_features(path)


def test_merge_and_validate_rejects_unmatched_images() -> None:
    features = _toy_features(["img_1", "img_2"])
    metadata = _toy_metadata(["img_1", "img_3"])

    with pytest.raises(DataValidationError, match="no metadata match"):
        merge_and_validate(features, metadata, enforce_contract=False)


def test_merge_and_validate_rejects_demographic_disagreement() -> None:
    features = _toy_features(["img_1", "img_2"])
    metadata = _toy_metadata(["img_1", "img_2"])
    metadata.loc[0, "sex"] = "female"

    with pytest.raises(DataValidationError, match="sex disagrees"):
        merge_and_validate(features, metadata, enforce_contract=False)


def test_melanoma_target_marks_only_mel() -> None:
    frame = pd.DataFrame(
        {"dx_original": ["mel", "nv", "bkl", "bcc", "akiec", "vasc", "df"]}
    )

    result, info = add_target(frame, "melanoma")

    assert result["target"].tolist() == [1, 0, 0, 0, 0, 0, 0]
    assert info["positive_classes"] == ["mel"]
    assert set(info["negative_classes"]) == {"nv", "bkl", "bcc", "akiec", "vasc", "df"}


def test_add_target_rejects_a_definition_that_misses_a_class() -> None:
    frame = pd.DataFrame({"dx_original": ["mel", "nv"]})
    with pytest.raises(ValueError, match="unknown target"):
        add_target(frame, "missing_task")


def test_split_is_group_disjoint_complete_and_deterministic() -> None:
    data = _toy_data()
    assigned = make_grouped_split(data, seed=42)
    summary = validate_split(assigned)

    assert set(assigned["split"]) == {"train", "validation", "test"}
    assert (assigned["fold"] >= 0).all()
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert summary["group_overlap"][f"{left}__{right}"] == 0

    again = make_grouped_split(data, seed=42)
    assert assigned["split"].tolist() == again["split"].tolist()
    assert assigned["fold"].tolist() == again["fold"].tolist()


def test_split_fingerprint_is_stable_and_sensitive_to_the_target() -> None:
    assigned = make_grouped_split(_toy_data(60), seed=42)

    first = split_fingerprint(assigned, "melanoma", 42)
    second = split_fingerprint(assigned.sample(frac=1, random_state=0), "melanoma", 42)
    other = split_fingerprint(assigned, "other", 42)

    assert first == second
    assert first != other


def test_validate_split_detects_lesion_leakage() -> None:
    assigned = make_grouped_split(_toy_data(60), seed=42)
    leaked = assigned.copy()
    lesion = leaked.loc[leaked["split"] == "train", "lesion_id"].iloc[0]
    leaked.loc[leaked["split"] == "test", "lesion_id"] = lesion

    with pytest.raises(SplitValidationError, match="overlap"):
        validate_split(leaked)


def test_load_modeling_frame_merges_features_and_assignments(tmp_path) -> None:
    image_ids = [f"img_{index}" for index in range(6)]
    features = _toy_features(image_ids)
    assignment = pd.DataFrame(
        {
            "image_id": image_ids,
            "lesion_id": [f"les_{index}" for index in range(6)],
            "target": [0, 1, 0, 1, 0, 1],
            "target_label": ["non_melanoma"] * 6,
            "fold": [0, 0, 1, 1, 2, 2],
            "split": ["test", "test", "validation", "validation", "train", "train"],
        }
    )
    interim_path = tmp_path / "interim.csv"
    split_path = tmp_path / "split.csv"
    features.to_csv(interim_path, index=False)
    assignment.to_csv(split_path, index=False)

    frame = load_modeling_frame(
        interim_path=interim_path, split_path=split_path
    )

    assert len(frame) == 6
    assert set(frame["split"]) == {"train", "validation", "test"}
    assert list(frame[list(config.ALL_FEATURES)].columns) == list(config.ALL_FEATURES)


@pytest.mark.integration
def test_real_ham10000_dataset_contract() -> None:
    if not config.FEATURES_PATH.exists() or not config.METADATA_PATH.exists():
        pytest.skip("raw HAM10000 inputs are not present")

    features = load_features()
    metadata = load_metadata()
    merged = merge_and_validate(features, metadata)

    assert len(merged) == config.EXPECTED_ROWS
    assert merged["lesion_id"].nunique() == config.EXPECTED_LESIONS
    assert set(merged["dx_original"].unique()) == set(config.EXPECTED_DX)

    with_target, _ = add_target(merged, "melanoma")
    assert int(with_target["target"].sum()) == 1_113
    assert int((with_target["target"] == 0).sum()) == 8_902
