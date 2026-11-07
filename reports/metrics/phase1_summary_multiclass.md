# Phase 1 — Data integrity and grouped split

- Generated at: `2026-11-06T10:55:12.136220+00:00`
- Target definition: **multiclass**
- Description: Original seven-class HAM10000 diagnosis task.
- Positive classes: None
- Negative classes: None
- Random seed: `42`
- Split fingerprint: `7a270d20b96c5b8609acafd68edab8c6b91e794674dc7e62a09ccb3e807c5e4f`

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
| train | 6008 | 4481 | 59.990% |
| validation | 2003 | 1494 | 20.000% |
| test | 2004 | 1495 | 20.010% |

### Target distribution by split

| Split | akiec | bcc | bkl | df | mel | nv | vasc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 195 | 308 | 659 | 69 | 669 | 4023 | 85 |
| validation | 66 | 103 | 220 | 23 | 222 | 1341 | 28 |
| test | 66 | 103 | 220 | 23 | 222 | 1341 | 29 |

### Original diagnosis distribution by split

| Split | akiec | bcc | bkl | df | mel | nv | vasc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 195 | 308 | 659 | 69 | 669 | 4023 | 85 |
| validation | 66 | 103 | 220 | 23 | 222 | 1341 | 28 |
| test | 66 | 103 | 220 | 23 | 222 | 1341 | 29 |

## Leakage checks

| Check | Result |
| --- | --- |
| Lesion overlap train/validation | 0 |
| Lesion overlap train/test | 0 |
| Lesion overlap validation/test | 0 |
| Max target prevalence difference | 0.0004 |
| Prevalence tolerance | 0.0300 |

## Outputs

| Artifact | Path |
| --- | --- |
| interim_dataset | data/interim/ham10000_with_lesions.csv |
| split_assignments | data/processed/split_assignments_multiclass.csv |
| manifest | reports/metrics/phase1_manifest_multiclass.json |
| summary | reports/metrics/phase1_summary_multiclass.md |

## Notes for later phases

- Use `split` for the final holdout.
- Run grouped cross-validation **inside `split == 'train'`** (recompute `StratifiedGroupKFold` on the training rows, or use the precomputed training folds 2, 3, and 4 as a 3-fold grouped CV). Never include validation or test rows in CV.
- `lesion_id` is a grouping key only. It must never be a model feature.
- `dx_original`, `dx_current`, `dx_type`, `target`, `target_label`, `fold`, and `split` are label/audit columns. They must never be model features.
- The Phase 1 default target is `melanoma` (melanoma vs all non-melanoma), matching the current notebooks. Use `--target malignant` or `--target multiclass` to regenerate the split for another task.

## Reproduce

```bash
uv run --python 3.13 --with pandas --with openpyxl --with scikit-learn==1.9.1 python scripts/build_phase1_dataset.py --target multiclass --seed 42 --n-splits 5 --test-fold 0 --val-fold 1
```
