# Phase 3 — Evaluation protocol

## Goal

Phase 3 makes model results interpretable and honest. It does **not** tune or
select a final model. The reference model is the untuned Phase 2 logistic
regression, compared against simple baselines on the validation split.

## Split roles

| Split | Use |
|---|---|
| `train` | Model fitting and grouped-CV threshold selection. |
| `validation` | One evaluation at the frozen threshold. |
| `test` | Not used for fitting, threshold selection, or metric reporting. |

The test split is reserved for the final model after model selection and
hyperparameter tuning are complete.

## Threshold protocol

1. Run grouped 5-fold CV (`StratifiedGroupKFold` by `lesion_id`) on `split == "train"`.
2. Generate out-of-fold positive-class probabilities.
3. Choose the F2-maximising threshold.
4. Freeze that threshold.
5. Apply it once to the validation split.

The report also shows a secondary high-sensitivity threshold chosen on the same
out-of-fold predictions, targeting sensitivity ≥ 0.90. Validation sensitivity can
differ from the OOF estimate; both are reported.

## Metrics

Class `1` is always the positive class: melanoma for `--target melanoma`,
malignant for `--target malignant`.

| Metric | Meaning |
|---|---|
| Sensitivity / recall | TP / (TP + FN) |
| Specificity | TN / (TN + FP) |
| PPV / precision | TP / (TP + FP) |
| NPV | TN / (TN + FN) |
| F1 | Harmonic mean of PPV and sensitivity |
| F2 | Recall-weighted harmonic mean; weights recall twice as much as precision |
| Balanced accuracy | Mean of sensitivity and specificity |
| Average precision / PR-AUC | Threshold-independent ranking quality for the positive class |
| ROC-AUC | Threshold-independent ranking quality across all thresholds |
| Brier score | Mean squared error of predicted probabilities; lower is better |

Accuracy is included in the machine-readable output but is not a headline
metric because the positive prevalence is only about 11%.

## Confidence intervals

The validation set contains images nested within lesions. Confidence intervals
therefore resample **whole lesions** with replacement, not individual images.
The default is 2,000 cluster-bootstrap replicates and a 95% percentile interval.
This avoids pretending that multiple images of the same lesion are independent
cases.

## Baselines

The report compares the reference model with:

- majority class / all-negative;
- uniform random scores;
- age-only logistic regression;
- metadata-only logistic regression (`age`, `sex`, `localization`);
- colour-only logistic regression;
- shape-only logistic regression;
- all-feature logistic regression (`k=15`).

All fitted baselines use the same preprocessing pipeline, grouped CV, and
threshold protocol. Feature-subset baselines keep all of their small feature
sets rather than applying `SelectKBest`.

## Usage

```bash
uv run --python 3.13 \
  --with pandas --with openpyxl --with scikit-learn==1.9.1 --with matplotlib \
  python scripts/run_phase3_evaluation.py --target melanoma
```

Outputs:

- `reports/metrics/phase3_metrics_<target>.json` — full metrics, CIs, thresholds, confusion matrices.
- `reports/metrics/phase3_summary_<target>.md` — human-readable report.
- `reports/figures/phase3_<target>_*.png` — PR, ROC, calibration, threshold sweep, and confusion-matrix figures.

Options:

```text
--target {melanoma,malignant}
--n-splits 5
--k 15
--bootstrap-reps 2000
--sensitivity 0.90
--figures / --no-figures
```

Multiclass evaluation is intentionally out of scope for Phase 3. It needs a
different metric and threshold protocol.

## Code

- `src/melanoma_detection/evaluate.py` — binary metrics, threshold selection, lesion-clustered bootstrap.
- `scripts/run_phase3_evaluation.py` — model/baseline orchestration, reports, and figures.
- `tests/test_evaluate.py` — metric correctness, threshold selection, bootstrap determinism.

## Limitations

- The reference model is untuned and is not a final model.
- The validation set is used for one final evaluation after OOF threshold selection.
- Bootstrap intervals reflect sampling uncertainty in the validation set, not model-selection uncertainty.
- HAM10000 has no patient identifier; lesion grouping is the best available leakage protection.
- These results are not clinical evidence and must not be presented as a diagnostic device.
