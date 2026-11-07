"""Tabular + embedding fusion pipeline.

All preprocessing, scaling, and PCA happen inside the scikit-learn pipeline, so
they are fitted on training folds only. That is the same leakage rule as the
tabular phases.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .. import config
from ..features import build_preprocessor


def build_fusion_pipeline(
    *,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    embedding_features: Sequence[str],
    n_components: int = config.FUSION_N_COMPONENTS,
    classifier: Any | None = None,
    seed: int = config.SEED,
) -> Pipeline:
    """Build ``preprocess -> PCA(embeddings) -> classifier`` as one pipeline."""
    numeric_features = tuple(numeric_features)
    categorical_features = tuple(categorical_features)
    embedding_features = tuple(embedding_features)
    if not embedding_features:
        raise ValueError("at least one embedding column is required")
    if n_components < 1:
        raise ValueError("n_components must be at least 1")

    tabular = build_preprocessor(
        scale_numeric=False,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    embedding_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=seed)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("tabular", tabular, list(numeric_features) + list(categorical_features)),
            ("embedding", embedding_pipeline, list(embedding_features)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    if classifier is None:
        classifier = ExtraTreesClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def fusion_feature_columns(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    embedding_features: Sequence[str],
) -> list[str]:
    return [*numeric_features, *categorical_features, *embedding_features]


def fusion_artifact_path(target: str = config.TARGET_DEFINITION) -> Path:
    return config.MODELS_DIR / f"fusion_{target}_{config.FUSION_MODEL_NAME}.joblib"


def fusion_pointer_path(target: str = config.TARGET_DEFINITION) -> Path:
    return config.MODELS_DIR / f"fusion_{target}_frozen.json"
