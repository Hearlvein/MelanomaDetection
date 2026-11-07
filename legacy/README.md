# Legacy sources (archived)

These files are the original project sources. They are kept for reference and
reproducibility, **not** as the supported pipeline.

> The tabular metrics produced by these notebooks are **not valid** for melanoma
> detection. `LabelEncoder` maps `mel=0` and `nomel=1`, so `pos_label=1` scores
> the non-melanoma class. The split is also image-level, so the same lesion can
> appear in both train and test.

## Contents

| Path | Origin |
|---|---|
| `notebooks/FeaturesExtraction.ipynb` | Feature extraction + custom CNN + VGG16 transfer learning |
| `notebooks/melanoma_detection.ipynb` | Tuned tabular models (pre-filled GridSearch results) |
| `notebooks/melanoma_detection_grid_searched.ipynb` | Tabular models with GridSearch cells enabled |
| `ml/melanoma_detection.py` | Script version of the tabular workflow |
| `README_upstream.md` | Original repository README |

## What the extraction notebook expects

```text
HAM10000_Dataset/
├── HAM10000/{image_id}.jpg
└── HAM10000_segmentations_lesion_tschandl/{image_id}_segmentation.png
```

The images and masks are not distributed with this repository. See
`data/README.md` for the expected layout and validation command.

## What was actually run

| Experiment | Test setup | Melanoma recall |
|---|---|---:|
| Gradient Boosting (tabular) | 80/20 image split | 0.02 |
| Custom CNN | 70/15/15 image split | 0.23 |
| VGG16 transfer | 70/15/15 image split | 0.29 |

The headline tabular F1 of 0.94 is the **non-melanoma** F1. The supported,
leak-free results live in `reports/metrics/`.

## Reproducing the legacy runs

The original assets (`.keras` models, PCA/t-SNE figures) can be fetched with:

```bash
python scripts/fetch_legacy_assets.py --all
```

The notebooks are not runnable as-is without the HAM10000 images and masks.
Paths in the notebooks are relative to the repository root.
