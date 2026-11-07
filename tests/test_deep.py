from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from melanoma_detection.deep import datasets, evaluate, train


def _make_images(tmp_path, count: int = 6, size: int = 24):
    image_dir = tmp_path / "HAM10000"
    image_dir.mkdir()
    rows = []
    rng = np.random.default_rng(0)
    for index in range(count):
        image_id = f"ISIC_{index:04d}"
        image = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
        Image.fromarray(image, mode="RGB").save(image_dir / f"{image_id}.jpg")
        target = index % 2
        rows.append(
            {
                "image_id": image_id,
                "lesion_id": f"HAM_{index:07d}",
                "target": target,
                "target_label": "melanoma" if target else "non_melanoma",
                "fold": index % 3,
                "split": "train" if index < 4 else "validation",
            }
        )
    return pd.DataFrame(rows), image_dir


def test_load_rgb_image_returns_the_requested_shape_and_range(tmp_path) -> None:
    frame, image_dir = _make_images(tmp_path, count=1)
    path = datasets.resolve_image_path(frame.loc[0, "image_id"], image_dir)
    image = datasets.load_rgb_image(path, image_size=(16, 12))
    assert image.shape == (12, 16, 3)  # (height, width, channels)
    assert image.dtype == np.float32
    assert 0.0 <= float(image.min()) and float(image.max()) <= 255.0


def test_iter_batches_respects_frame_order(tmp_path) -> None:
    frame, image_dir = _make_images(tmp_path, count=6)
    batches = list(
        datasets.iter_batches(
            frame,
            image_dir,
            image_size=(16, 16),
            batch_size=4,
            shuffle=False,
        )
    )
    assert [len(targets) for _, targets in batches] == [4, 2]
    assert np.array_equal(
        np.concatenate([targets for _, targets in batches]), frame["target"].to_numpy(dtype=float)
    )


def test_augmentation_is_deterministic_for_a_seed() -> None:
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    image = np.arange(32 * 32 * 3, dtype=np.float32).reshape(32, 32, 3)
    assert np.array_equal(
        datasets.augment_image(image.copy(), rng_a),
        datasets.augment_image(image.copy(), rng_b),
    )


def test_missing_images_are_reported(tmp_path) -> None:
    frame, image_dir = _make_images(tmp_path, count=2)
    (image_dir / "ISIC_0001.jpg").unlink()
    assert datasets.missing_images(frame, image_dir) == ["ISIC_0001"]
    with pytest.raises(FileNotFoundError, match="missing"):
        datasets.assert_images_exist(frame, image_dir)


def test_balanced_class_weight_is_inverse_frequency() -> None:
    weights = train.balanced_class_weight(np.array([0, 0, 0, 1]))
    assert weights[0] == pytest.approx(4 / 6)
    assert weights[1] == pytest.approx(4 / 2)


def test_deep_test_guard_refuses_to_overwrite(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("melanoma_detection.config.METRICS_DIR", tmp_path)
    metrics_path, _ = evaluate.deep_test_paths("melanoma", "custom_cnn")
    metrics_path.write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        evaluate.assert_deep_test_not_yet_evaluated("melanoma", "custom_cnn")

    forced = evaluate.assert_deep_test_not_yet_evaluated(
        "melanoma", "custom_cnn", force=True
    )
    assert forced == metrics_path


def test_custom_cnn_trains_predicts_and_round_trips(tmp_path) -> None:
    tf = pytest.importorskip("tensorflow")
    from melanoma_detection.deep import models

    frame, image_dir = _make_images(tmp_path, count=6)
    train_frame = frame.loc[frame["split"] == "train"].reset_index(drop=True)
    validation_frame = frame.loc[frame["split"] == "validation"].reset_index(drop=True)

    image_size = (24, 24)
    train_dataset = datasets.make_tf_dataset(
        train_frame, image_dir, image_size=image_size, batch_size=2, shuffle=True
    )
    validation_dataset = datasets.make_tf_dataset(
        validation_frame, image_dir, image_size=image_size, batch_size=2
    )

    model = models.build_model("custom_cnn", input_shape=(24, 24, 3), base_filters=2)
    models.compile_model(model)
    history = train.train_model(
        model,
        train_dataset,
        validation_dataset,
        epochs=1,
        class_weight=train.class_weight_for(train_frame),
        verbose=0,
    )
    assert "loss" in history.history

    scores = evaluate.predict_scores(model, validation_dataset)
    assert scores.shape == (len(validation_frame),)
    assert np.all((scores >= 0) & (scores <= 1))

    artifact, sidecar = train.save_deep_artifact(
        model,
        {"target": "melanoma", "model": "custom_cnn", "seed": 42},
        tmp_path / "model.keras",
    )
    assert artifact.exists()
    assert sidecar.exists()

    loaded, metadata = train.load_deep_artifact(artifact)
    assert metadata["target"] == "melanoma"
    loaded_scores = evaluate.predict_scores(loaded, validation_dataset)
    assert np.allclose(scores, loaded_scores, atol=1e-5)

    extractor = models.build_embedding_extractor(loaded)
    embeddings = extractor.predict(validation_dataset, verbose=0)
    assert embeddings.shape == (len(validation_frame), 8)


def test_vgg16_transfer_builds_and_exposes_embeddings(tmp_path) -> None:
    pytest.importorskip("tensorflow")
    from melanoma_detection.deep import models

    model = models.build_model(
        "vgg16", input_shape=(32, 32, 3), weights=None, dense_units=8
    )
    assert model.output_shape == (None, 1)
    assert model.get_layer("embedding").units == 8

    embedding_model = models.build_vgg16_embedding_model(
        input_shape=(32, 32, 3), weights=None
    )
    assert embedding_model.output_shape == (None, 512)
