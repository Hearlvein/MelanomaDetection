# Phase 3 — Evaluation protocol and baselines

- Generated at: `2026-11-06T13:56:12.769297+00:00`
- Target: **malignant**
- Positive class: **malignant** (`target = 1`)
- Negative class: **benign** (`target = 0`)
- Description: Malignant vs benign lesions (melanoma + BCC + AKIEC vs the rest).
- Validation threshold source: `grouped CV OOF predictions on split == train`
- Bootstrap: 2000 lesion-clustered replicates, 95% percentile intervals
- Secondary sensitivity target: 0.90

## Protocol

- Model fitting, preprocessing, feature selection, and threshold selection use **training data only**.
- Threshold selection uses grouped cross-validation out-of-fold predictions inside `split == "train"`.
- The validation split is evaluated at the frozen threshold; it is not used for repeated tuning.
- The test split is **not used for fitting, threshold selection, or metric reporting**.
- Confidence intervals resample whole `lesion_id` groups, not individual images.

## Data roles

| Split | Rows | Lesions | Positive | Negative | Prevalence | Used for |
| --- | --- | --- | --- | --- | --- | --- |
| train | 6009 | 4482 | 1172 | 4837 | 0.195 | fit + grouped-CV threshold selection |
| validation | 2003 | 1494 | 391 | 1612 | 0.195 | frozen-threshold evaluation |
| test | 2003 | 1494 | 391 | 1612 | 0.195 | not used for modelling |

## Models and feature sets

| Model | Numeric features | Categorical features | Description |
| --- | --- | --- | --- |
| majority | 0 | 0 | Constant zero score; classifies everything as negative. |
| random | 0 | 0 | Uniform random scores; empirical chance baseline. |
| age_only | 1 | 0 | Logistic regression using age only. |
| metadata_only | 1 | 2 | Age, sex, and anatomical localization only. |
| colour_only | 12 | 0 | RGB channels and colour proportions only. |
| shape_only | 4 | 0 | Asymmetry, perimeter, and area only. |
| all_features | 17 | 2 | Phase 2 reference: all features, logistic regression, k=15. |

## Threshold selection

| Model | F2 threshold (OOF) | Sensitivity ≥ 0.90 threshold (OOF) | Validation prevalence |
| --- | --- | --- | --- |
| majority | 0.500 | — | 0.195 |
| random | 0.500 | — | 0.195 |
| age_only | 0.471 | 0.320 | 0.195 |
| metadata_only | 0.399 | 0.296 | 0.195 |
| colour_only | 0.298 | 0.385 | 0.195 |
| shape_only | 0.348 | 0.379 | 0.195 |
| all_features | 0.406 | 0.295 | 0.195 |

## Validation metrics at the frozen F2 threshold

| Model | Threshold | Sensitivity | Specificity | PPV | NPV | F1 | F2 | Predicted positive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| majority | 0.500 | 0.000 [0.000–0.000] | 1.000 [1.000–1.000] | — | 0.805 [0.781–0.828] | 0.000 [0.000–0.000] | 0.000 [0.000–0.000] | 0.000 |
| random | 0.500 | 0.545 [0.498–0.592] | 0.506 [0.481–0.530] | 0.211 [0.181–0.242] | 0.821 [0.792–0.848] | 0.304 [0.267–0.340] | 0.414 [0.373–0.451] | 0.504 |
| age_only | 0.471 | 0.760 [0.697–0.818] | 0.635 [0.606–0.664] | 0.335 [0.292–0.376] | 0.916 [0.891–0.939] | 0.465 [0.419–0.509] | 0.606 [0.556–0.655] | 0.442 |
| metadata_only | 0.399 | 0.854 [0.799–0.905] | 0.597 [0.567–0.628] | 0.339 [0.300–0.381] | 0.944 [0.921–0.965] | 0.486 [0.440–0.529] | 0.655 [0.608–0.698] | 0.491 |
| colour_only | 0.298 | 0.951 [0.921–0.976] | 0.424 [0.397–0.454] | 0.286 [0.253–0.320] | 0.973 [0.955–0.987] | 0.440 [0.399–0.480] | 0.649 [0.611–0.685] | 0.649 |
| shape_only | 0.348 | 0.921 [0.890–0.947] | 0.220 [0.199–0.240] | 0.222 [0.196–0.250] | 0.919 [0.888–0.947] | 0.358 [0.323–0.394] | 0.566 [0.527–0.603] | 0.808 |
| all_features | 0.406 | 0.839 [0.785–0.890] | 0.653 [0.626–0.683] | 0.369 [0.328–0.412] | 0.943 [0.921–0.962] | 0.513 [0.469–0.557] | 0.669 [0.625–0.712] | 0.443 |

## Threshold-independent metrics

| Model | Average precision | ROC-AUC | Brier score |
| --- | --- | --- | --- |
| majority | 0.195 [0.172–0.219] | 0.500 [0.500–0.500] | 0.195 [0.172–0.219] |
| random | 0.209 [0.181–0.243] | 0.531 [0.500–0.560] | 0.324 [0.311–0.336] |
| age_only | 0.364 [0.315–0.421] | 0.743 [0.705–0.780] | 0.204 [0.193–0.216] |
| metadata_only | 0.439 [0.382–0.512] | 0.799 [0.768–0.830] | 0.187 [0.175–0.200] |
| colour_only | 0.340 [0.292–0.399] | 0.747 [0.718–0.776] | 0.210 [0.198–0.221] |
| shape_only | 0.327 [0.279–0.383] | 0.677 [0.641–0.709] | 0.233 [0.227–0.240] |
| all_features | 0.474 [0.411–0.547] | 0.817 [0.786–0.848] | 0.179 [0.166–0.191] |

## Confusion matrices at the frozen F2 threshold

| Model | TP | FP | FN | TN |
| --- | --- | --- | --- | --- |
| majority | 0 | 0 | 391 | 1612 |
| random | 213 | 797 | 178 | 815 |
| age_only | 297 | 589 | 94 | 1023 |
| metadata_only | 334 | 650 | 57 | 962 |
| colour_only | 372 | 928 | 19 | 684 |
| shape_only | 360 | 1258 | 31 | 354 |
| all_features | 328 | 560 | 63 | 1052 |

## Validation at the OOF high-sensitivity threshold (target ≥ 0.90)

The threshold is selected on training OOF predictions. Validation sensitivity can differ.

| Model | Threshold | Sensitivity | Specificity | PPV | NPV | Predicted positive |
| --- | --- | --- | --- | --- | --- | --- |
| age_only | 0.320 | 0.905 | 0.345 | 0.251 | 0.938 | 0.704 |
| metadata_only | 0.296 | 0.921 | 0.458 | 0.292 | 0.960 | 0.616 |
| colour_only | 0.385 | 0.898 | 0.499 | 0.303 | 0.953 | 0.578 |
| shape_only | 0.379 | 0.908 | 0.263 | 0.230 | 0.922 | 0.770 |
| all_features | 0.295 | 0.898 | 0.523 | 0.313 | 0.955 | 0.559 |

## Figures

- `reports/figures/phase3_malignant_precision_recall.png`
- `reports/figures/phase3_malignant_roc.png`
- `reports/figures/phase3_malignant_calibration.png`
- `reports/figures/phase3_malignant_threshold_sweep.png`
- `reports/figures/phase3_malignant_confusion_matrix.png`

## Interpretation notes

- The all-features model is the untuned Phase 2 reference, not a final selected model.
- Baselines use the same split and metrics; they are not tuned on validation.
- A PR-AUC close to the validation prevalence means the model is near chance.
- The test split must remain untouched until the model and threshold are frozen after model selection.
- These results are not clinical evidence and must not be presented as a diagnostic device.

## Reproduce

```bash
uv run --python 3.13 --with pandas --with openpyxl --with scikit-learn==1.9.1 --with matplotlib python scripts/run_phase3_evaluation.py --target malignant --n-splits 5 --k 15 --bootstrap-reps 2000 --sensitivity 0.9
```
