# movingDataset

Synthetic vehicle dataset generation utilities for small road-scene and dark-intersection experiments.

This repository is intended to store the reproducible parts of the work:

- dataset generation scripts in `scripts/`
- experiment configs in `configs/`
- lightweight preview artifacts in `box_target_examples/` and `viz_dark_intersection_32k/`
- loader helpers for downstream training code

Generated datasets are intentionally ignored by Git because they can contain thousands of PNG files. Regenerate them from the scripts and configs after cloning on a new machine.

## Setup

Use Python 3.10+ with `numpy` and `Pillow` available.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install numpy pillow
```

## Generate a Dark-Intersection Dataset

Most recent dark-intersection variants are driven by JSON configs:

```powershell
python scripts\generate_dark_intersection_dataset.py --config configs\dark_intersection_32k_spread.json --output dataset_dark_intersection_32k_spread
```

Many configs keep the original remote output path used during experiments. Use `--output` when generating locally so the data is written inside the current checkout or another local dataset directory.

## Useful Checks

Compile the Python scripts:

```powershell
python -m py_compile scripts\generate_dataset.py scripts\generate_dark_intersection_dataset.py scripts\nearest_train_baseline.py scripts\make_box_target_examples.py
```

Measure split leakage with the nearest-train coordinate baseline:

```powershell
python scripts\nearest_train_baseline.py dataset_dark_intersection_10k_spatial_split_3px
```

## Repository Hygiene

Keep generated dataset directories out of Git. Commit source, configs, validation scripts, and small previews only. If a generated dataset is needed on another machine, regenerate it from the committed config or copy it outside Git.
