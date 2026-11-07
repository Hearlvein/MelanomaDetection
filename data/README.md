# Data layout

```text
data/
├── raw/                  tracked inputs
│   ├── HAM10000_metadata.csv          official 7-class HAM10000 metadata
│   └── HAM10000_features_couleurs.xlsx  V1 tabular extract (21 columns)
├── reference/            tracked reference outputs, used for parity checks
│   └── HAM10000_features_couleurs_v2.csv  V2 extract (23 columns)
├── external/             NOT tracked; images and masks live here
│   └── HAM10000_Dataset/
│       ├── HAM10000/{image_id}.jpg
│       └── HAM10000_segmentations_lesion_tschandl/{image_id}_segmentation.png
├── interim/              generated; not tracked
└── processed/            generated splits; not tracked
```

## What is V1 vs V2?

- **V1** (`data/raw/HAM10000_features_couleurs.xlsx`): 21 columns, the extract
  used by the legacy notebooks and by the frozen Phase 5 melanoma model. It has
  no `circularity` or `eccentricity`.
- **V2** (`data/reference/HAM10000_features_couleurs_v2.csv`): 23 columns, the
  newer extract that adds `circularity` and `eccentricity`.

The frozen artifact stays on V1. V2 is the input for the new extraction line and
should be regenerated from images with:

```bash
python scripts/extract_features.py --overwrite
```

`scripts/extract_features.py` keeps `lesion_id` (24 columns). The archived
upstream CSV has 23 columns because the notebook dropped `lesion_id` before
extraction. Both are accepted by the loader; when the extracted file is absent,
`--features v2` falls back to the archived reference CSV so the pipeline can be
tested without images.

## Building a dataset and split

V1 (default) and V2 write separate artifacts:

```bash
# Frozen Phase 5 input (21 columns)
python scripts/build_phase1_dataset.py --target melanoma --features v1

# New extraction line (adds circularity + eccentricity)
python scripts/build_phase1_dataset.py --target melanoma --features v2
```

| Version | Interim dataset | Split assignments |
|---|---|---|
| `v1` | `data/interim/ham10000_with_lesions.csv` | `data/processed/split_assignments_{target}.csv` |
| `v2` | `data/interim/ham10000_with_lesions_v2.csv` | `data/processed/split_assignments_{target}_v2.csv` |

V1 is never overwritten by a V2 build. The split fingerprint is identical across
versions because it depends only on `image_id`, `lesion_id`, and `target`.

## External images and masks

The HAM10000 images and the Tschandl lesion segmentation masks are not part of
this repository. Download them from their original sources and place them under
`data/external/HAM10000_Dataset/` exactly as shown above.

Validate the layout before running extraction or the CNN:

```bash
python scripts/check_external_data.py
```

The check verifies that every `image_id` in `data/raw/HAM10000_metadata.csv` has
both an image and a mask. It does not download anything.

## Embeddings and fusion

The late-fusion line writes an embedding table and manifest to:

```text
data/interim/embeddings_<target>_<model>.npz   image_id + embedding matrix
reports/metrics/embeddings_<target>_<model>.json
```

The `.npz` file stores `image_ids`, a float32 `embeddings` matrix, the model
name, and the target. The manifest records dimensions, hashes, counts, the
split fingerprint, and the extraction command. The fusion model itself lives
under `models/fusion_<target>_extra_trees.{joblib,json}` with a frozen pointer.
See `docs/fusion.md`.

## Manifest discipline

Every generated table should be accompanied by a manifest containing:

- the input hashes;
- the row, image, and lesion counts;
- the feature schema version;
- the package versions;
- the command that produced it.

`scripts/extract_features.py` writes this manifest automatically.
