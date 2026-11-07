# Phase 2 — Leak-free pipeline smoke check

## Scope

Phase 2 builds the reusable modelling pipeline and proves that preprocessing and
feature selection run inside the training fold. It does **not** tune models or
select a final threshold.

## Pipeline

```text
ColumnTransformer
  numeric:     median imputation -> StandardScaler
  categorical: most-frequent imputation -> OneHotEncoder(handle_unknown="ignore")
SelectKBest(f_classif, k=15)
LogisticRegression(class_weight="balanced")
```

- Features: `config.ALL_FEATURES` (35 columns before encoding).
- Label/audit columns (`image_id`, `lesion_id`, `dx_original`, `dx_current`,
  `dx_type`, `target`, `target_label`, `fold`, `split`) are never features.
- Imputation, scaling, encoding, and selection are fitted by the pipeline inside
  each CV training fold.
- Grouped 5-fold CV (`StratifiedGroupKFold`, seed 42) runs only on rows where
  `split == "train"`. Validation and test rows never enter CV.

## Results

| Target | CV metric | Fold scores | Mean (std) | Selected features |
|---|---|---|---:|---:|
| `melanoma` | average precision | 0.3900, 0.3250, 0.3403, 0.3060, 0.3182 | 0.3359 (0.0292) | 15 of 35 |
| `malignant` | average precision | see JSON | 0.4980 (0.0530) | 15 of 35 |
| `multiclass` | balanced accuracy | see JSON | 0.5594 (0.0290) | 15 of 35 |

Machine-readable results are in `phase2_smoke_<target>.json`.

These are smoke-test numbers. They are untuned, use a single linear baseline, and
must not be reported as final model performance.

## Data used

| Target | Train rows | Train lesions | Validation rows | Validation lesions |
|---|---:|---:|---:|---:|
| `melanoma` | 6,009 | 4,482 | 2,003 | 1,494 |
| `malignant` | 6,009 | 4,482 | 2,003 | 1,494 |
| `multiclass` | 6,008 | 4,481 | 2,003 | 1,494 |

The multiclass split differs by one row because stratification is target-specific.

## Reproduce

```bash
uv run --python 3.13 --with pandas --with openpyxl --with scikit-learn==1.9.1 \
  python scripts/check_phase2_pipeline.py --target melanoma
```

## Tests

```text
18 passed
```

The suite covers target construction, lesion-grouped splits, grouped CV,
preprocessor statistics learned from train only, unknown-category handling,
optional scaling, selector behaviour, and a real-data pipeline smoke path.

## Next

Phase 3 is implemented in `src/melanoma_detection/evaluate.py` and
`scripts/run_phase3_evaluation.py`. It adds validation-only evaluation,
lesion-clustered confidence intervals, baselines, and OOF threshold selection.
See `reports/metrics/phase3_summary_melanoma.md`.

Model comparison, hyperparameter tuning, and the single final test evaluation
belong to later phases.
