from __future__ import annotations

import pytest

from melanoma_detection import config
from melanoma_detection.data import add_target, load_features, load_metadata, merge_and_validate


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
