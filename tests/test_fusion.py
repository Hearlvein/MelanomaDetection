from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from melanoma_detection import config
from melanoma_detection.data import DataValidationError
from melanoma_detection.fusion.embeddings import (
    embedding_feature_names,
    load_embeddings,
    merge_embeddings,
    save_embeddings,
)
from melanoma_detection.fusion.evaluate import (
    assert_fusion_test_not_yet_evaluated,
    fusion_test_paths,
)
from melanoma_detection.fusion.pipeline import build_fusion_pipeline, fusion_feature_columns
from melanoma_detection.fusion.train import fit_fusion


def _synthetic_frame(n_rows: int = 40, n_embeddings: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    numeric = list(config.get_numeric_features("v2"))
    categorical = list(config.get_categorical_features("v2"))
    train_end = int(n_rows * 0.6)
    validation_end = int(n_rows * 0.8)

    rows = []
    for index in range(n_rows):
        if index < train_end:
            split = "train"
        elif index < validation_end:
            split = "validation"
        else:
            split = "test"
        target = index % 2
        row: dict[str, object] = {
            "image_id": f"ISIC_{index:04d}",
            "lesion_id": f"HAM_{index:07d}",
            "target": target,
            "target_label": "melanoma" if target else "non_melanoma",
            "fold": index % config.N_SPLITS,
            "split": split,
        }
        for column in numeric:
            row[column] = float(rng.normal())
        for column in categorical:
            row[column] = "male" if column == "sex" else "back"
        for column in embedding_feature_names(n_embeddings):
            row[column] = float(rng.normal() + 0.5 * target)
        rows.append(row)
    return pd.DataFrame(rows)


def test_embedding_table_roundtrip_and_merge(tmp_path) -> None:
    frame = _synthetic_frame(n_rows=6, n_embeddings=3).drop(
        columns=list(embedding_feature_names(3))
    )
    image_ids = frame["image_id"].to_numpy()
    vectors = np.arange(18, dtype=np.float32).reshape(6, 3)

    path, manifest = save_embeddings(
        tmp_path / "embeddings.npz",
        image_ids,
        vectors,
        model_name="vgg16",
        target="melanoma",
        manifest_path=tmp_path / "embeddings.json",
    )
    loaded_ids, loaded_vectors = load_embeddings(path)

    assert np.array_equal(loaded_ids, image_ids)
    assert np.allclose(loaded_vectors, vectors)
    assert manifest.exists()

    merged = merge_embeddings(frame, loaded_ids, loaded_vectors)
    for position, column in enumerate(embedding_feature_names(3)):
        assert np.allclose(merged[column].to_numpy(), vectors[:, position])
    assert merged["image_id"].tolist() == frame["image_id"].tolist()


def test_merge_embeddings_rejects_missing_rows() -> None:
    frame = _synthetic_frame(n_rows=4, n_embeddings=2).drop(
        columns=list(embedding_feature_names(2))
    )
    image_ids = frame["image_id"].to_numpy()[:-1]
    vectors = np.zeros((len(image_ids), 2), dtype=np.float32)

    with pytest.raises(DataValidationError, match="no embedding"):
        merge_embeddings(frame, image_ids, vectors)


def test_fusion_pipeline_contains_embedding_pca_inside_preprocessor() -> None:
    pipeline = build_fusion_pipeline(
        numeric_features=["age"],
        categorical_features=["sex"],
        embedding_features=list(embedding_feature_names(5)),
        n_components=3,
    )
    preprocessor = pipeline.named_steps["preprocessor"]
    embedding_step = {
        name: transformer for name, transformer, _ in preprocessor.transformers
    }["embedding"]

    assert embedding_step.named_steps["scaler"].__class__.__name__ == "StandardScaler"
    assert embedding_step.named_steps["pca"].n_components == 3
    assert pipeline.named_steps["classifier"].__class__.__name__ == "ExtraTreesClassifier"


def test_fit_fusion_freezes_threshold_and_reports_validation() -> None:
    frame = _synthetic_frame(n_rows=40, n_embeddings=4, seed=7)
    # Make the label easier to learn so the synthetic threshold is meaningful.
    for column in embedding_feature_names(4):
        frame[column] = frame[column] + 2.0 * frame["target"]
    frame = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)
    frame["split"] = "train"
    frame.loc[:23, "split"] = "train"
    frame.loc[24:31, "split"] = "validation"
    frame.loc[32:, "split"] = "test"

    from sklearn.ensemble import ExtraTreesClassifier

    classifier = ExtraTreesClassifier(
        n_estimators=20,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=7,
        n_jobs=1,
    )
    feature_columns = fusion_feature_columns(
        list(config.get_numeric_features("v2")),
        list(config.get_categorical_features("v2")),
        list(embedding_feature_names(4)),
    )
    assert len(feature_columns) == len(set(feature_columns))

    pipeline, result, columns = fit_fusion(
        frame,
        numeric_features=list(config.get_numeric_features("v2")),
        categorical_features=list(config.get_categorical_features("v2")),
        embedding_features=list(embedding_feature_names(4)),
        n_components=2,
        classifier=classifier,
        n_splits=3,
        seed=7,
        bootstrap_reps=5,
    )

    assert columns == feature_columns
    assert 0.0 <= result["threshold_selection"]["f2_threshold"] <= 1.0
    assert set(result["validation"]["metrics"]) >= {
        "sensitivity",
        "specificity",
        "f2",
        "average_precision",
        "roc_auc",
    }
    assert result["n_components"] == 2
    assert result["embedding_features"] == list(embedding_feature_names(4))

    scores = pipeline.predict_proba(frame.loc[frame["split"] == "validation", columns])
    assert scores.shape == (8, 2)
    assert np.isfinite(scores).all()


def test_fusion_test_guard_requires_force(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "METRICS_DIR", tmp_path)
    metrics_path, _ = fusion_test_paths("melanoma")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        assert_fusion_test_not_yet_evaluated("melanoma")
    assert assert_fusion_test_not_yet_evaluated("melanoma", force=True) == metrics_path
