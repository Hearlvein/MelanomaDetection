# Melanoma Detection

Reproducible analysis of the HAM10000-derived feature dataset.

## Status

Phases 1–5 are implemented: lesion-grouped data splits, a leak-free modelling
pipeline, a validation-only evaluation protocol with baselines and
lesion-clustered confidence intervals, an explicit imbalance comparison, and
model tuning with a frozen artifact and a one-shot test evaluation.

Three model lines now share that protocol:

- **tabular** — V1 feature workbook → leak-free scikit-learn pipelines;
- **deep** — Keras/TensorFlow CNNs on raw images;
- **fusion** — V2 tabular features + frozen image embeddings → ExtraTrees.

Only `melanoma` receives official Phase 4/5, deep, fusion, and one-shot test
runs. The original notebooks are kept for reference but their result tables are
**not valid** for melanoma detection: they score the benign class by default and
split images instead of lesions.

## Project layout

```text
data/
  raw/          V1 workbook and official HAM10000 metadata
  reference/    archived V2 extract for parity checks
  external/     NOT tracked: HAM10000 images and lesion masks
  interim/      merged data, V2 extract, embedding tables
  processed/    split assignments and active-target pointer
src/melanoma_detection/
  config.py     paths, constants, feature versions, target specifications
  schema.py     canonical V1/V2 column contracts
  data.py       loading, merge, validation, target construction
  splits.py     lesion-grouped split and grouped CV
  features.py   leak-free preprocessor and feature selector
  pipeline.py   preprocess -> select -> classify pipeline
  evaluate.py   binary metrics, thresholds, lesion-clustered bootstrap
  training.py   shared OOF/threshold/evaluation helpers
  imbalance.py  class-weight and resampling strategies
  train.py      tuning, threshold freezing, artifacts, one-shot guard
  extraction/   V2 colour/shape feature extraction from images and masks
  deep/         Keras datasets, models, training, evaluation, embeddings
  fusion/       embedding storage, PCA pipeline, fusion training/evaluation
scripts/
  build_phase1_dataset.py       build V1 or V2 dataset and split
  extract_features.py           V2 image feature extraction
  check_phase2_pipeline.py
  run_phase3_evaluation.py
  run_phase4_imbalance.py
  run_phase5_model_selection.py
  evaluate_final_test.py
  train_deep.py                 train a Keras classifier
  evaluate_deep.py              one-shot deep test evaluation
  extract_embeddings.py         frozen image embeddings for fusion
  train_fusion.py               V2 + embedding fusion model
  evaluate_fusion.py            one-shot fusion test evaluation
tests/
docs/
models/
reports/metrics/
reports/figures/
```

## Quick start

Install the package and the optional lines you need:

```bash
# tabular + extraction + tests
pip install -e ".[tabular,extraction,dev]"

# add TensorFlow/Keras for the deep and fusion lines
pip install -e ".[tabular,extraction,deep,dev]"
```

The equivalent Make targets are `make install` and `make install-deep`. The
legacy `requirements*.txt` files are kept for convenience; `pyproject.toml` is
the canonical dependency definition.

Build the default melanoma-vs-non-melanoma split:

```bash
uv run --python 3.13 --with pandas --with openpyxl --with scikit-learn==1.9.1 \
  python scripts/build_phase1_dataset.py --target melanoma
```

Equivalent plain-Python command after installing the requirements:

```bash
python scripts/build_phase1_dataset.py --target melanoma
```

## Target options

- `melanoma` — melanoma vs all non-melanoma (default; current notebook task).
- `malignant` — melanoma + BCC + AKIEC vs benign.
- `multiclass` — original seven HAM10000 diagnoses.

See `docs/target_definition.md`.

## Feature versions

Two tabular schemas are supported and must not be mixed:

| Version | Columns | Use |
|---|---:|---|
| V1 | 21 | Frozen Phase 5 ExtraTrees artifact and the legacy pipeline |
| V2 | 23 legacy / 24 with `lesion_id` | New extraction, deep/fusion tabular inputs |

V1 is frozen: never add features to it. New features (notably `circularity` and
`eccentricity`) belong to V2. Both versions produce the same lesion-grouped
split fingerprint because the split depends only on `image_id`, `lesion_id`, and
`target`. See `data/README.md` for paths and build commands.

## Phase 1 outputs

- `data/interim/ham10000_with_lesions.csv` — merged dataset with `lesion_id`, original diagnosis, and target.
- `data/processed/split_assignments_<target>.csv` — `image_id`, `lesion_id`, target, fold, and split.
- `data/processed/active_target.txt` — current target, split file, and fingerprint.
- `reports/metrics/phase1_manifest_<target>.json` — hashes, software versions, counts, and validation results.
- `reports/metrics/phase1_summary_<target>.md` — human-readable audit report.

## Phase 2 pipeline

`src/melanoma_detection/features.py` and `pipeline.py` provide a leak-free
`ColumnTransformer` + `SelectKBest` + classifier pipeline:

- numeric: median imputation, optional standardisation;
- categorical: most-frequent imputation, `OneHotEncoder(handle_unknown="ignore")`;
- feature selection and model fitting happen inside the training fold only.

```python
from melanoma_detection.data import load_modeling_frame
from melanoma_detection.pipeline import build_pipeline

frame = load_modeling_frame("melanoma")
train = frame[frame["split"] == "train"]
pipeline = build_pipeline(classifier, scale_numeric=True, k=15)
```

Run the Phase 2 smoke check (grouped CV on the training split only):

```bash
uv run --python 3.13 --with pandas --with openpyxl --with scikit-learn==1.9.1 \
  python scripts/check_phase2_pipeline.py --target melanoma
```

This writes `reports/metrics/phase2_smoke_<target>.json`. It is a pipeline
smoke test, not a final model result. Binary targets use average precision; the
multiclass target uses balanced accuracy.

Current smoke results (grouped 5-fold CV on `split == "train"` only,
`LogisticRegression(class_weight="balanced")`, `k=15`):

| Target | Mean CV score (std) |
|---|---:|
| `melanoma` | 0.3359 AP (0.0292) |
| `malignant` | 0.4980 AP (0.0530) |
| `multiclass` | 0.5594 balanced accuracy (0.0290) |

See `docs/pipeline.md` for details.

## Phase 3 evaluation

Phase 3 adds a validation-only evaluation protocol:

- class `1` is the positive class; metrics are never scored on the benign class;
- thresholds are selected from grouped-CV out-of-fold predictions inside `split == "train"`;
- the validation split is evaluated once at the frozen threshold;
- the test split is not used for fitting, threshold selection, or metric reporting;
- confidence intervals resample whole `lesion_id` groups, not images;
- baselines include majority, random, age-only, metadata-only, colour-only, and shape-only models.

```bash
uv run --python 3.13 \
  --with pandas --with openpyxl --with scikit-learn==1.9.1 --with matplotlib \
  python scripts/run_phase3_evaluation.py --target melanoma
```

Outputs:

- `reports/metrics/phase3_metrics_<target>.json`
- `reports/metrics/phase3_summary_<target>.md`
- `reports/figures/phase3_<target>_*.png`

`--target malignant` is supported. Multiclass evaluation needs a separate
protocol and is intentionally out of scope for Phase 3. See
`docs/evaluation.md`.

## Phase 4 imbalance comparison

Phase 4 compares class weighting and resampling under the same grouped-CV and
threshold protocol:

- strategies: `none`, `balanced`, `balanced_subsample`, `random_over`,
  `random_under`, `smote`;
- models: Logistic Regression, Random Forest, HistGradientBoosting;
- resampling happens inside the training fold only, after preprocessing and
  feature selection;
- the validation split is used for comparison; the test split is not used.

```bash
uv run --python 3.13 \
  --with pandas --with openpyxl --with scikit-learn==1.9.1 --with imbalanced-learn \
  python scripts/run_phase4_imbalance.py --target melanoma
```

Outputs:

- `reports/metrics/phase4_imbalance_<target>.json`
- `reports/metrics/phase4_imbalance_<target>.md`

See `docs/imbalance.md`.

## Phase 5 model selection and final test

Phase 5 tunes a small model set with grouped CV inside `split == "train"`,
compares the tuned candidates on validation, freezes one model and one
threshold, and then evaluates the test split once.

Step 1 — select and freeze (validation only):

```bash
uv run --python 3.13 \
  --with pandas --with openpyxl --with scikit-learn==1.9.1 --with imbalanced-learn \
  python scripts/run_phase5_model_selection.py --target melanoma
```

This writes:

- `reports/metrics/phase5_model_selection_<target>.json`
- `reports/metrics/phase5_model_selection_<target>.md`
- `models/phase5_<target>_<model>.joblib` and `.json`
- `models/phase5_<target>_frozen.json`

Step 2 — evaluate the test split exactly once, only after confirming the target
and operating point:

```bash
uv run --python 3.13 \
  --with pandas --with openpyxl --with scikit-learn==1.9.1 \
  --with imbalanced-learn --with matplotlib \
  python scripts/evaluate_final_test.py --target melanoma
```

The final-test script refuses to overwrite an existing report unless `--force`
is passed. See `docs/model_selection.md` and `docs/model_card.md`.

### Final melanoma result (test evaluated once)

The confirmed final target is `melanoma`. The frozen model is
**ExtraTrees** with `class_weight="balanced"` and threshold **0.220**.

| Metric | Test result (n=2,003; 1,494 lesions) |
|---|---:|
| Sensitivity | 0.834 [0.784–0.884] |
| Specificity | 0.696 [0.671–0.721] |
| PPV | 0.256 [0.212–0.299] |
| NPV | 0.971 [0.960–0.981] |
| F2 | 0.574 [0.517–0.622] |
| Average precision | 0.391 [0.311–0.470] |
| ROC-AUC | 0.835 [0.806–0.862] |
| Brier score | 0.092 [0.084–0.101] |

At the high-sensitivity operating point (threshold 0.134): sensitivity 0.928,
specificity 0.518, PPV 0.194.

The test AP is higher than the validation AP (0.359), but that is sampling
variation across lesion-disjoint splits, not evidence that the model improved.

Reports:

- `reports/metrics/phase5_final_test_melanoma.json`
- `reports/metrics/phase5_final_test_melanoma.md`
- `reports/figures/phase5_final_test_melanoma_*.png`

**The test split has been evaluated once. Do not rerun it without an explicit,
documented `--force` decision.**

## Image feature extraction (V2)

The supported extraction replaces the legacy notebook cells and keeps
`lesion_id` so the lesion-grouped split can be enforced:

```bash
python scripts/build_phase1_dataset.py --target melanoma --features v2
python scripts/extract_features.py --overwrite
```

This requires HAM10000 images and Tschandl lesion masks under
`data/external/HAM10000_Dataset/`; validate them with
`python scripts/check_external_data.py`. When the generated V2 CSV is absent,
the loader falls back to `data/reference/HAM10000_features_couleurs_v2.csv`,
which is sufficient for tabular-only tests. See `data/README.md`.

## Deep learning line

`src/melanoma_detection/deep/` contains Keras datasets, models, training,
evaluation, and penultimate embedding extraction. Two architectures are
available: `custom_cnn` and a frozen ImageNet `vgg16` transfer model.

```bash
python scripts/train_deep.py --target melanoma --model custom_cnn
python scripts/evaluate_deep.py --target melanoma --model custom_cnn
python scripts/extract_embeddings.py --target melanoma --model vgg16 --weights imagenet
```

Training reads only train/validation rows. Test evaluation is guarded by
`reports/metrics/deep_final_test_<target>_<model>.json` and refuses to overwrite
without `--force`. Outputs are `models/deep_<target>_<model>.keras` plus a JSON
sidecar and pointer, and reports under `reports/metrics/`.

Only `melanoma` is officially supported. Real image runs require the external
images; synthetic checks use `--weights none`. See `docs/deep_learning.md`.

## Late fusion

Fusion combines V2 tabular features with frozen image embeddings:

```bash
# 1. build V2 split and extract embeddings (once)
python scripts/build_phase1_dataset.py --target melanoma --features v2
python scripts/extract_embeddings.py --target melanoma --features v2 --weights imagenet

# 2. train and evaluate once
python scripts/train_fusion.py --target melanoma --features v2
python scripts/evaluate_fusion.py --target melanoma --features v2
```

Defaults are ExtraTrees, VGG16 pooled embeddings, and PCA to 32 components.
Imputation, standardisation, and PCA are fitted inside the scikit-learn pipeline
on training folds only. The grouped-OOF F2 threshold is frozen in the artifact;
the test report is guarded exactly like the tabular and deep lines.

Outputs are `data/interim/embeddings_<target>_<model>.npz` with a manifest,
`models/fusion_<target>_extra_trees.joblib` plus sidecar and pointer, and
`reports/metrics/fusion_*.json` / `.md`. Only `melanoma` is officially
supported. See `docs/fusion.md`.

## Tests

```bash
# after installing the package and dev extras
pytest -q

# or with uv, including TensorFlow
uv run --python 3.13 --with pytest --with pandas --with openpyxl \
  --with scikit-learn==1.9.1 --with imbalanced-learn --with scikit-image \
  --with pillow --with tqdm --with tensorflow-cpu pytest -q
```

Deep tests call `pytest.importorskip("tensorflow")`, so a tabular-only environment
runs the rest of the suite and skips them. The real-data integration test is
skipped automatically if the raw inputs are absent. CI runs a tabular job and a
TensorFlow job; see `.github/workflows/tests.yml`.

## Phase 1 acceptance

- 10,015 rows and 7,470 unique lesions after merging.
- Zero lesion overlap between train, validation, and test.
- Target prevalence approximately preserved across splits.
- Reproducible split fingerprint.
- Label/audit columns excluded from model features.

## Status of the phases

- Phase 1: data integrity and lesion-grouped splits — complete.
- Phase 2: leak-free preprocessing and pipeline — complete.
- Phase 3: validation-only evaluation, baselines, threshold selection, CIs — complete.
- Phase 4: class-weight and resampling comparison — complete.
- Phase 5: tuning, validation comparison, frozen artifact, one-shot test — complete for `melanoma`.
- Extraction: V1/V2 schemas and image feature extraction — complete; real V2 extraction needs external images and masks.
- Deep line: Keras training, evaluation, and embedding extraction — implemented and smoke-tested; real runs need external images.
- Fusion line: V2 tabular + embedding pipeline, training, and one-shot evaluation — implemented and smoke-tested; real runs need external images.

The `melanoma` tabular test split has been evaluated once. No real deep or fusion
test report exists yet because HAM10000 images are not in the repository. The
`malignant` target has not received a final test evaluation and would need its
own Phase 4/5 run first. Do not rerun the melanoma test to “try another
threshold”.

Do not report model metrics from the legacy notebooks as final results.

`data/raw` is the source of truth for the new pipeline. The root-level
`HAM10000_features_couleurs.xlsx` is kept only because the legacy notebooks still
reference it; keep the two copies in sync or archive the legacy notebooks.

For model development, run grouped cross-validation **inside the training split**
(e.g. `StratifiedGroupKFold` on rows where `split == "train"`). The `fold` column
is the global 5-fold assignment: training rows are folds 2–4, validation is fold
1, and test is fold 0. Do not run CV over validation or test rows.
