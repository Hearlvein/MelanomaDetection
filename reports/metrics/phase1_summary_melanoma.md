# Phase 1 — Data integrity and grouped split

- Generated at: `2026-11-06T10:55:19.417008+00:00`
- Target definition: **melanoma**
- Description: Melanoma vs all non-melanoma diagnoses (current notebook task).
- Positive classes: ['mel']
- Negative classes: ['nv', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
- Random seed: `42`
- Split fingerprint: `b39eae73a410580368397df3c203b644d7887084cea27627084a318a5cffa862`

## Inputs

| File | SHA-256 |
| --- | --- |
| data/raw/HAM10000_features_couleurs.xlsx | cb2b56ce02c0869000b3766c8b9abf2bfea6bdf7f6a465f02242b313c041a3b4 |
| data/raw/HAM10000_metadata.csv | 06aee5e603d313ba69de38263053c792a1dff030eb9c44a96ceb477ceefb99a5 |

## Dataset checks

| Check | Result |
| --- | --- |
| Rows | 10015 |
| Unique image IDs | 10015 |
| Unique lesions | 7470 |
| Original diagnosis classes | akiec, bcc, bkl, df, mel, nv, vasc |
| Missing values | {'age': 57} |

## Split summary

| Split | Rows | Lesions | Share |
| --- | --- | --- | --- |
| train | 6009 | 4482 | 60.000% |
| validation | 2003 | 1494 | 20.000% |
| test | 2003 | 1494 | 20.000% |

### Target distribution by split

| Split | non_melanoma | melanoma |
| --- | --- | --- |
| train | 5341 | 668 |
| validation | 1781 | 222 |
| test | 1780 | 223 |

### Original diagnosis distribution by split

| Split | akiec | bcc | bkl | df | mel | nv | vasc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 194 | 283 | 697 | 75 | 668 | 4023 | 69 |
| validation | 67 | 111 | 190 | 14 | 222 | 1373 | 26 |
| test | 66 | 120 | 212 | 26 | 223 | 1309 | 47 |

## Leakage checks

| Check | Result |
| --- | --- |
| Lesion overlap train/validation | 0 |
| Lesion overlap train/test | 0 |
| Lesion overlap validation/test | 0 |
| Max target prevalence difference | 0.0003 |
| Prevalence tolerance | 0.0300 |

## Outputs

| Artifact | Path |
| --- | --- |
| interim_dataset | data/interim/ham10000_with_lesions.csv |
| split_assignments | data/processed/split_assignments_melanoma.csv |
| manifest | reports/metrics/phase1_manifest_melanoma.json |
| summary | reports/metrics/phase1_summary_melanoma.md |

## Notes for later phases

- Use `split` for the final holdout.
- Run grouped cross-validation **inside `split == 'train'`** (recompute `StratifiedGroupKFold` on the training rows, or use the precomputed training folds 2, 3, and 4 as a 3-fold grouped CV). Never include validation or test rows in CV.
- `lesion_id` is a grouping key only. It must never be a model feature.
- `dx_original`, `dx_current`, `dx_type`, `target`, `target_label`, `fold`, and `split` are label/audit columns. They must never be model features.
- The Phase 1 default target is `melanoma` (melanoma vs all non-melanoma), matching the current notebooks. Use `--target malignant` or `--target multiclass` to regenerate the split for another task.

## Reproduce

```bash
uv run --python 3.13 --with pandas --with openpyxl --with scikit-learn==1.9.1 python scripts/build_phase1_dataset.py --target melanoma --seed 42 --n-splits 5 --test-fold 0 --val-fold 1
```
