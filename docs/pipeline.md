# Phase 2 — Leak-free pipeline

## Design

The pipeline has three stages and always receives **raw feature columns**:

```text
ColumnTransformer
  numeric:     median imputation -> optional StandardScaler
  categorical: most-frequent imputation -> OneHotEncoder(handle_unknown="ignore")
SelectKBest(f_classif, k=15)
Classifier
```

All three stages are fitted inside the training fold. The following are intentionally **not** done anywhere:

- `fit_transform` on the full dataset before a split;
- `SelectKBest.fit` on the full training set before `GridSearchCV`;
- scaling or encoding the validation/test set with statistics learned from those rows;
- `cross_val_score` on data that already includes validation/test rows.

## Feature columns

Defined in `src/melanoma_detection/config.py`:

- `NUMERIC_FEATURES`: age, colour channels/proportions, asymmetry, perimeter, area.
- `CATEGORICAL_FEATURES`: sex, localization.
- `ALL_FEATURES`: the only columns passed to the model.
- `LABEL_COLUMNS`: image/lesion identifiers, diagnoses, target, fold, split. Never features.

`build_preprocessor()` and `build_pipeline()` also accept explicit
`numeric_features` and `categorical_features` sequences. This is used by the
Phase 3 age-only, metadata-only, colour-only, and shape-only baselines without
weakening the default full-feature schema.

## Usage

```python
from melanoma_detection.data import load_modeling_frame
from melanoma_detection.pipeline import build_pipeline
from melanoma_detection.splits import make_grouped_cv

frame = load_modeling_frame("melanoma")
train = frame[frame["split"] == "train"]

pipeline = build_pipeline(
    classifier,
    scale_numeric=True,   # use False for tree models
    k=15,
)
cv = make_grouped_cv(n_splits=5, seed=42)

# raw X, y, and groups go directly into CV
scores = cross_val_score(
    pipeline,
    train[list(config.ALL_FEATURES)],
    train["target"],
    groups=train[config.GROUP_COLUMN],
    cv=cv,
    scoring="average_precision",
)
```

## Validation / test rules

- Hyperparameter tuning and threshold selection happen inside `split == "train"` using grouped CV.
- The validation split may be used for a final model/threshold choice after CV, but not for repeated tuning.
- The test split is touched once, after the model and threshold are frozen.
- The precomputed `fold` column is a global 5-fold assignment. Training rows are folds 2–4, validation is fold 1, test is fold 0.

## Smoke check

```bash
uv run --python 3.13 --with pandas --with openpyxl --with scikit-learn==1.9.1 \
  python scripts/check_phase2_pipeline.py --target melanoma
```

This runs grouped CV **on the training split only** and writes
`reports/metrics/phase2_smoke_<target>.json`. It is a pipeline smoke test, not a
final model result. Binary targets are scored with average precision; the
multiclass target is scored with balanced accuracy because average precision is
a binary-ranking metric.

## Phase 2 smoke results

All runs use `LogisticRegression(class_weight="balanced")`, `k=15`, and grouped
5-fold CV on `split == "train"` only.

| Target | CV metric | Mean (std) | Selected features |
|---|---|---:|---:|
| `melanoma` | average precision | 0.3359 (0.0292) | 15 of 35 |
| `malignant` | average precision | 0.4980 (0.0530) | 15 of 35 |
| `multiclass` | balanced accuracy | 0.5594 (0.0290) | 15 of 35 |

These are smoke-test numbers to prove the pipeline runs without leakage. They are
not tuned and must not be reported as final results. Phase 3 will add proper
model comparison, threshold selection, and melanoma-focused metrics.
