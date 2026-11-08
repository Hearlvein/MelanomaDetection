# Results

## Problem

HAM10000 contains 10,015 dermatoscopic images from 7,470 lesions. The task is
binary: identify **melanoma** (`dx = mel`) against every other diagnosis. The
positive class is rare (11.1% of images), so accuracy is misleading and the main
metrics are average precision (AP), ROC-AUC, sensitivity, specificity, and F2.

## Protocol

1. Merge the colour/shape workbook with the official metadata.
2. Create a reproducible 60/20/20 split with all images of a lesion in the same
   split (`StratifiedGroupKFold`, seed 42). Fingerprint:
   `b39eae73a410580368397df3c203b644d7887084cea27627084a318a5cffa862`.
3. Tune models with grouped cross-validation inside the training split only.
4. Select the decision threshold from out-of-fold training scores (maximise F2).
5. Compare already-tuned models on validation and freeze one.
6. Evaluate the frozen model and threshold on the test split **once**.

The test split is never used for fitting, tuning, feature selection, or
threshold selection. Confidence intervals resample whole lesions, not images.

## Dataset

| Split | Rows | Lesions | Melanoma | Non-melanoma | Prevalence |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 6,009 | 4,482 | 668 | 5,341 | 0.111 |
| validation | 2,003 | 1,494 | 222 | 1,781 | 0.111 |
| test | 2,003 | 1,494 | 223 | 1,780 | 0.111 |

No lesion appears in more than one split.

## Baselines (validation)

| Model | Features | AP | ROC-AUC | F2 |
| --- | --- | ---: | ---: | ---: |
| Majority | none | 0.111 | 0.500 | 0.000 |
| Random | none | 0.107 | 0.495 | 0.282 |
| Age only | age | 0.220 | 0.690 | 0.441 |
| Metadata only | age, sex, localization | 0.255 | 0.731 | 0.455 |
| Colour only | RGB channels and proportions | 0.165 | 0.703 | 0.508 |
| Shape only | asymmetry, perimeter, area | 0.188 | 0.673 | 0.437 |
| All features (LR) | all tabular features | 0.299 | 0.792 | 0.537 |

Metadata and colour contain signal, but neither is sufficient on its own.

## Tuned models (validation)

Hyperparameters were tuned with grouped 5-fold CV inside the training split.
The table shows the best imbalance strategy retained for each family. Thresholds
come from training out-of-fold scores.

| Model | Strategy | CV AP | Validation AP | Validation F2 | Threshold |
| --- | --- | ---: | ---: | ---: | ---: |
| Logistic Regression | none | 0.341 | 0.305 | 0.538 | 0.090 |
| Random Forest | random undersampling | 0.316 | 0.355 | 0.502 | 0.422 |
| HistGradientBoosting | balanced | 0.368 | 0.353 | 0.533 | 0.153 |
| ExtraTrees | balanced | 0.348 | 0.359 | 0.533 | 0.220 |

The full 4 models x 4 strategies sweep is in `model_comparison.json`.

## Frozen model

**ExtraTrees**, `class_weight="balanced"`, with:

- `n_estimators=300`, `max_depth=30`, `min_samples_leaf=4`
- `max_features=None`, `SelectKBest(k=30)`
- Validation AP 0.359, ROC-AUC 0.820
- Frozen F2 threshold **0.220**
- High-sensitivity threshold **0.134** (target sensitivity 0.90)

Artifact: `models/extra_trees_melanoma.joblib`.
Sidecar: `models/extra_trees_melanoma.json`.
Pointer: `models/extra_trees_melanoma_frozen.json`.

## Final test (evaluated once)

Test data: 2,003 images, 1,494 lesions, 223 melanoma, prevalence 0.111.

| Metric | Value (95% CI) |
| --- | --- |
| Sensitivity | 0.834 [0.784–0.884] |
| Specificity | 0.696 [0.671–0.721] |
| PPV | 0.256 [0.212–0.299] |
| NPV | 0.971 [0.960–0.981] |
| F1 | 0.392 [0.336–0.441] |
| F2 | 0.574 [0.517–0.622] |
| Average precision | 0.391 [0.311–0.470] |
| ROC-AUC | 0.835 [0.806–0.862] |
| Brier score | 0.092 [0.084–0.101] |

Confusion matrix at threshold 0.220: TN 1,239, FP 541, FN 37, TP 186.

At the high-sensitivity threshold 0.134: sensitivity 0.928, specificity 0.518,
PPV 0.194, NPV 0.983.

Validation AP was 0.359 and test AP 0.391. The small increase is sampling
variation, not evidence that the test split is easier.

## Figures

- `reports/figures/final_test_melanoma_precision_recall.png`
- `reports/figures/final_test_melanoma_roc.png`
- `reports/figures/final_test_melanoma_calibration.png`
- `reports/figures/final_test_melanoma_confusion_matrix.png`

## Limitations

- HAM10000 is a dermatoscopic dataset; these are pre-extracted tabular features,
  not a raw-image model.
- No patient identifier is available. Lesion grouping is the strongest leakage
  control available, but it is not perfect.
- The dataset is not representative of all skin tones, populations, or clinical
  workflows.
- Validation was used for model selection, so its estimate is mildly optimistic.
  The test estimate is the one to report.
- Calibration is imperfect, especially after class weighting.
- This is a school project, not a diagnostic device. It must not guide patient
  care.

## Reproduce

```bash
python scripts/01_dataset.py
python scripts/02_compare_models.py
# Do not rerun step 3: the one-shot test result is already recorded.
python scripts/04_deep_learning.py check-images
python scripts/04_deep_learning.py train --model custom_cnn
```
