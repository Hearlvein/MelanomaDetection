from __future__ import annotations

from collections.abc import Sequence

from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def build_numeric_pipeline(*, scale_numeric: bool = True) -> Pipeline:
    """Impute numeric features, optionally standardise them.

    Tree models do not need scaling, so ``scale_numeric=False`` keeps their
    pipeline simpler. Linear models and SVMs should use the default ``True``.
    """
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def build_categorical_pipeline() -> Pipeline:
    """Impute categorical features and one-hot encode them.

    ``handle_unknown='ignore'`` means a category seen only in validation or test
    is encoded as all zeros instead of raising an error. The encoder's category
    list is learned from the training fold only.
    """
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )


def build_preprocessor(
    *,
    scale_numeric: bool = True,
    numeric_features: Sequence[str] | None = None,
    categorical_features: Sequence[str] | None = None,
) -> ColumnTransformer:
    """Build the leak-free feature preprocessor.

    The preprocessor is deliberately unaware of labels. It must be fitted inside
    a training fold; never call ``fit`` or ``fit_transform`` on the full dataset.

    ``numeric_features`` and ``categorical_features`` default to the full schema
    and can be overridden for baseline/ablation models.
    """
    numeric = tuple(config.NUMERIC_FEATURES if numeric_features is None else numeric_features)
    categorical = tuple(
        config.CATEGORICAL_FEATURES if categorical_features is None else categorical_features
    )
    if not numeric and not categorical:
        raise ValueError("at least one numeric or categorical feature is required")

    transformers = []
    if numeric:
        transformers.append(
            ("num", build_numeric_pipeline(scale_numeric=scale_numeric), list(numeric))
        )
    if categorical:
        transformers.append(("cat", build_categorical_pipeline(), list(categorical)))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_selector(k: int | str | None = config.DEFAULT_SELECT_K) -> SelectKBest | str:
    """Return a univariate selector or passthrough.

    ``k`` can be an integer, ``"all"``, or ``None``. The selector is fitted inside
    each training fold, so its scores never see validation/test labels.
    """
    if k is None:
        return "passthrough"
    return SelectKBest(score_func=f_classif, k=k)
