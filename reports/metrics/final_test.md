# Final test evaluation

- Generated at: `2026-11-06T22:22:48.781696+00:00`
- Target: **melanoma**
- Positive class: **melanoma** (`target = 1`)
- Negative class: **non_melanoma** (`target = 0`)
- Model: **ExtraTrees**
- Imbalance strategy: `balanced`
- Frozen F2 threshold: 0.220
- Frozen artifact: `models/extra_trees_melanoma.joblib`
- Split fingerprint: `b39eae73a410580368397df3c203b644d7887084cea27627084a318a5cffa862`

> This report evaluates the test split **once**. The model and threshold were
> frozen before this script ran. Do not rerun without an explicit `--force`.

## Test data

| Rows | Lesions | Positive | Negative | Prevalence |
| --- | --- | --- | --- | --- |
| 2003 | 1494 | 223 | 1780 | 0.111 |

## Test metrics at the frozen threshold

| Sensitivity | Specificity | PPV | NPV | F1 | F2 | AP | ROC-AUC | Brier | Predicted positive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.834 [0.784–0.884] | 0.696 [0.671–0.721] | 0.256 [0.212–0.299] | 0.971 [0.960–0.981] | 0.392 [0.336–0.441] | 0.574 [0.517–0.622] | 0.391 [0.311–0.470] | 0.835 [0.806–0.862] | 0.092 [0.084–0.101] | 0.363 |

## Confusion matrix

| TP | FP | FN | TN |
| --- | --- | --- | --- |
| 186 | 541 | 37 | 1239 |

## Validation vs test

Validation was the model-selection split, so its estimate is mildly optimistic.
The test estimate below is the one to report as the unbiased estimate.

| Metric | Validation | Test (95% CI) |
| --- | --- | --- |
| Average precision | 0.359 | 0.391 [0.311–0.470] |
| ROC-AUC | 0.820 | 0.835 [0.806–0.862] |
| Sensitivity | 0.784 | 0.834 [0.784–0.884] |
| Specificity | 0.681 | 0.696 [0.671–0.721] |
| PPV | 0.234 | 0.256 [0.212–0.299] |
| F2 | 0.533 | 0.574 [0.517–0.622] |
| Brier score | 0.095 | 0.092 [0.084–0.101] |

## High-sensitivity operating point

| Threshold | Sensitivity | Specificity | PPV | NPV | Predicted positive |
| --- | --- | --- | --- | --- | --- |
| 0.134 | 0.928 | 0.518 | 0.194 | 0.983 | 0.532 |

## Figures

- `reports/figures/final_test_melanoma_precision_recall.png`
- `reports/figures/final_test_melanoma_roc.png`
- `reports/figures/final_test_melanoma_calibration.png`
- `reports/figures/final_test_melanoma_confusion_matrix.png`

## Model card summary

- **Intended use:** research evaluation of melanoma detection on HAM10000 tabular features.
- **Not for clinical use:** this is not a diagnostic device and must not guide patient care.
- **Target definition:** Melanoma vs all non-melanoma diagnoses.
- **Features:** 19 tabular metadata, colour, and shape features.
- **Training rows:** 6009 images from 4482 lesions.
- **Validation rows:** 2003 images from 1494 lesions.
- **Grouping:** lesions are disjoint across train/validation/test; confidence intervals resample lesions.
- **Threshold:** selected from grouped-CV out-of-fold predictions on the training split.

## Limitations

- HAM10000 is a dermatoscopic image dataset; the model uses pre-extracted tabular features, not raw pixels.
- No patient identifier is available, so lesion grouping is the strongest available leakage control.
- The dataset is not representative of all skin tones, populations, or clinical workflows.
- Validation was used for model selection; test metrics are the unbiased estimate but still reflect one dataset.
- Probability calibration may be imperfect, especially after class weighting or resampling.
- Performance at this operating point is not sufficient for clinical deployment.

## Reproduce

```bash
uv run --python 3.13 --with pandas --with openpyxl --with scikit-learn==1.9.1 --with imbalanced-learn --with matplotlib python scripts/03_final_test.py --target melanoma --artifact models/extra_trees_melanoma.joblib --bootstrap-reps 2000
```
