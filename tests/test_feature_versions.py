from __future__ import annotations

import pandas as pd
import pytest

from melanoma_detection import config
from melanoma_detection.data import (
    DataValidationError,
    detect_feature_version,
    load_features,
    merge_and_validate,
)
from melanoma_detection.schema import (
    FEATURE_COLUMNS_V1,
    FEATURE_COLUMNS_V2,
    FEATURE_COLUMNS_V2_LEGACY,
)


def _row(columns: tuple[str, ...], **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {column: 0.0 for column in columns}
    row.update(
        {
            "image_id": "ISIC_0000001",
            "lesion_id": "HAM_0000001",
            "dx": "mel",
            "age": 50.0,
            "sex": "male",
            "localization": "back",
        }
    )
    row.update(overrides)
    return row


def _metadata(**overrides: object) -> pd.DataFrame:
    row = {
        "lesion_id": "HAM_0000001",
        "image_id": "ISIC_0000001",
        "dx": "mel",
        "dx_type": "histo",
        "age": 50.0,
        "sex": "male",
        "localization": "back",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_detect_feature_version_with_and_without_lesion_id() -> None:
    assert detect_feature_version(pd.DataFrame(columns=list(FEATURE_COLUMNS_V1))) == "v1"
    assert detect_feature_version(pd.DataFrame(columns=list(FEATURE_COLUMNS_V2))) == "v2"
    assert (
        detect_feature_version(pd.DataFrame(columns=list(FEATURE_COLUMNS_V2_LEGACY)))
        == "v2_legacy"
    )


def test_detect_feature_version_rejects_extra_columns() -> None:
    columns = [*FEATURE_COLUMNS_V1, "leaked_label"]
    with pytest.raises(DataValidationError, match="unexpected"):
        detect_feature_version(pd.DataFrame(columns=columns))


def test_merge_verifies_and_drops_v2_lesion_id() -> None:
    features = pd.DataFrame([_row(FEATURE_COLUMNS_V2)])
    merged = merge_and_validate(features, _metadata(), enforce_contract=False)

    assert len(merged) == 1
    # lesion_id comes from the authoritative metadata, not from the feature table.
    assert merged.loc[0, "lesion_id"] == "HAM_0000001"
    assert "circularity" in merged.columns
    assert "eccentricity" in merged.columns


def test_merge_rejects_a_mismatched_v2_lesion_id() -> None:
    features = pd.DataFrame([_row(FEATURE_COLUMNS_V2, lesion_id="HAM_9999999")])
    with pytest.raises(DataValidationError, match="lesion_id disagrees"):
        merge_and_validate(features, _metadata(), enforce_contract=False)


def test_merge_accepts_a_v2_table_without_lesion_id() -> None:
    features = pd.DataFrame([_row(FEATURE_COLUMNS_V2_LEGACY)])
    merged = merge_and_validate(features, _metadata(), enforce_contract=False)
    assert len(merged) == 1
    assert merged.loc[0, "lesion_id"] == "HAM_0000001"


@pytest.mark.integration
def test_reference_v2_extract_loads_and_merges() -> None:
    reference = config.FEATURES_V2_REFERENCE_PATH
    if not reference.exists() or not config.METADATA_PATH.exists():
        pytest.skip("reference V2 extract or metadata is not present")

    features = load_features(reference, feature_version="v2")
    assert features.attrs["feature_version"] == "v2"
    assert features.attrs["feature_schema"] == "v2_legacy"
    assert len(features) == config.EXPECTED_ROWS
    assert "circularity" in features.columns
    assert "eccentricity" in features.columns

    merged = merge_and_validate(features, load_metadata_or_skip())
    assert len(merged) == config.EXPECTED_ROWS
    assert merged["lesion_id"].nunique() == config.EXPECTED_LESIONS


def load_metadata_or_skip() -> pd.DataFrame:
    # Imported lazily so the unit tests above stay independent of the raw data.
    from melanoma_detection.data import load_metadata

    return load_metadata()


@pytest.mark.integration
def test_v2_split_matches_the_frozen_v1_fingerprint() -> None:
    from melanoma_detection.data import add_target, load_features, load_metadata
    from melanoma_detection.splits import make_grouped_split, split_fingerprint

    if not config.FEATURES_PATH.exists() or not config.FEATURES_V2_REFERENCE_PATH.exists():
        pytest.skip("V1 workbook or V2 reference extract is not present")

    fingerprints: dict[str, str] = {}
    metadata = load_metadata()
    for version in ("v1", "v2"):
        features = load_features(feature_version=version)
        merged = merge_and_validate(features, metadata)
        with_target, _ = add_target(merged, "melanoma")
        assignment = make_grouped_split(
            with_target,
            target_col="target",
            group_col="lesion_id",
            n_splits=config.N_SPLITS,
            test_fold=config.TEST_FOLD,
            val_fold=config.VAL_FOLD,
            seed=config.SEED,
        )
        fingerprints[version] = split_fingerprint(assignment, "melanoma", config.SEED)

    # V2 only changes features; it must never change which lesion is in which split.
    assert fingerprints["v1"] == fingerprints["v2"]
