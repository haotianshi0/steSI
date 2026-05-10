# Spatial transcriptome and epitranscriptome: spatial imputation

Computational artifact for spatial imputation in spatial transcriptome and epitranscriptome data.

steSI stands for "Spatial Transcriptome and Epitranscriptome: Spatial Imputation".

Artifact type: computational model and computational analysis.

## Overview

This project evaluates whether spatial and neural imputation models can recover held-out A-to-I editing ratios for observed spot-site pairs. The prediction target is:

```text
editing_ratio = G / (A + G)
```

The benchmark is designed around a strict missingness convention:

```text
NaN = not observed
0.0 = observed, with zero or near-zero editing signal
```

This distinction is enforced throughout mask generation, model training, and metric computation. In particular, only truly observed entries are split into training and holdout sets; originally unobserved entries are not used for metric scoring.

## For Markers

Recommended files to inspect, in reading order:

End-to-end walkthrough

- `artifact_walkthrough.ipynb` - eleven-section walkthrough with embedded benchmark figures, smoke-test cell, and synthesis observations.

Reproducibility entry points

- `README.md` - installation, usage, and project overview (this file).
- `REPRODUCIBILITY.md` - exact commands to reproduce every committed result.
- `environment.yml` / `requirements.txt` - pinned dependencies.

Pre-rendered benchmark outputs

- `results/notebook_demo/` - three figures (mask-ratio line plot, method-comparison boxplot, 3x3 high-signal block heatmap) plus the two summary CSVs they were rendered from.

Project models

- `src/models/AIRGate-ST/train.py` - two-stage gate-and-ratio model (project's primary model).
- `src/models/AIRDiff-ST/train.py` - conditional diffusion-style ratio imputation model.
- `src/models/AIRGate-ST/shared/data_utils.py` - shared data loader, KNN graph, and plotting helpers used by every model and the visualization scripts.

Evaluation and visualization

- `evaluation/collect_layered_metrics.py` - layered metric computation with NaN/zero validation.
- `visualization/method_comparison_high_signal_block.py` - script behind the 3x3 high-signal heatmap in notebook Section 7.
- `visualization/plot_method_comparison_boxplots.py` - script behind the boxplot in notebook Section 6.
- `visualization/plot_selected_models_mask_ratio.py` - script behind the line plot in notebook Section 5.

Attribution and AI use

- `ATTRIBUTION.md` - original components and open-source dependencies.
- `AI_USE_DECLARATION.md` - AI assistance declaration.

Recommended sanity command (no GPU, no training):

```powershell
python experiments\run_harder_benchmark.py --sample_id 151673 --seed 1531 --mask_seed 1531 --model_seed 1531 --epochs 1 --device cpu --models all --dry_run --skip_dual_block
```

## Workflow

```text
h5ad dataset
  -> depth/site filtering
  -> observed_mask
  -> train_mask + val_mask
  -> model prediction matrices
  -> layered metrics
  -> reproducible figures
```

Core mask contract:

```python
not np.any(train_mask & val_mask)
np.array_equal(train_mask | val_mask, observed_mask)
```

## Models

| Model | Category | Role in the benchmark |
| --- | --- | --- |
| Mean Baseline | Statistical baseline | Per-site mean imputation. |
| SoftImpute | Matrix baseline | Low-rank matrix completion baseline. |
| Multiple Imputation | Statistical baseline | Iterative chained imputation baseline. |
| Spatial KNN | Spatial baseline | Neighbor-based spatial smoothing. |
| Spatial IDW | Spatial baseline | Inverse-distance spatial interpolation. |
| AIRGate-ST | Project model | Two-stage spatial gate-and-ratio imputation model. |
| AIRDiff-ST | Project model | Conditional diffusion-style ratio imputation model. |

The final reporting workflow uses 20%, 40%, and 60% holdout ratios.

## Repository Layout

```text
steSI/
+-- configs/          # example benchmark configuration
+-- data/             # bundled h5ad datasets
+-- evaluation/       # layered metrics and paired significance tests
+-- experiments/      # orchestration scripts
+-- results/          # generated locally; the only committed subfolder is
|                     # results/notebook_demo/, which holds the figures and
|                     # summary CSVs that artifact_walkthrough.ipynb displays
+-- src/
|   +-- masks/        # shared train/holdout mask generation
|   +-- models/       # model training entry points
|   |   +-- AIRGate-ST/shared/  # data loading, KNN graph, and heatmap helpers
|   |   |                       # imported by every model, the mask generator,
|   |   |                       # baselines, and the dual-block figure script
|   +-- shared/       # mask + baseline helpers, inner-mask split utility
+-- visualization/    # reproducible figure scripts
```

The project uses two helper folders with distinct scopes:

- `src/shared/` - `baseline_utils.py` (depth filtering, channel-wise spatial KNN/IDW, randomized-SVD SoftImpute) and `inner_mask_utils.py` (training-side inner validation split). These are used by `src/masks/generate_masks.py`, `src/models/baselines/train.py`, and both neural training scripts.

- `src/models/AIRGate-ST/shared/data_utils.py` - `.h5ad` loading, site filtering, KNN graph construction, train/val mask handling, prediction validation, layered metrics, and heatmap/diagnostic plotting. This module is imported by every model entry point (AIRGate-ST, AIRDiff-ST, baselines), by `src/masks/generate_masks.py`, and by `visualization/dual_block_observed_heatmap.py`. Each importing script adds `src/models/AIRGate-ST/` to `sys.path` and imports through a stable `from shared.data_utils import ...` line, so callers do not depend on AIRGate-ST's training internals.

`configs/artifact_summary.json` is a compact metadata file that summarizes the model count, retained model names, mask semantics, prediction semantics, and seed semantics. It is not the execution entry point; use the commands below or the protocol in `REPRODUCIBILITY.md` to run experiments.

## Installation

Recommended environment: Python 3.11 with CUDA-enabled PyTorch. CPU execution is supported but slower for neural models.

Open a terminal and move into the project root before running any command. The project root is the folder that contains `README.md`, `environment.yml`, `requirements.txt`, `src/`, `experiments/`, and `data/`.

Example:

```powershell
cd "C:\path\to\steSI"
```

Create the Conda environment:

```powershell
conda env create -f environment.yml
conda activate st_ai_impute
```

Alternatively, install dependencies inside an existing environment:

```powershell
python -m pip install -r requirements.txt
```

## Notebook Walkthrough

An executable notebook is provided at:

```text
artifact_walkthrough.ipynb
```

To open it from the project root:

```powershell
jupyter lab artifact_walkthrough.ipynb
```

The notebook walks through eleven short sections: project root resolution, data
availability check, a dry-run smoke test, the main benchmark command (off by
default), three benchmark-result figures (mask-ratio line plot, method-comparison
boxplot, and a 3x3 high-signal block heatmap), a synthesis of model-evaluation
observations grounded in those figures, a section explaining why prediction
`.npy` files are not committed, model interpretation prose, and a metric
interpretation guide. The figures and their summary CSVs are loaded from
`results/notebook_demo/` (described below); prediction `.npy` files are not
committed and would be regenerated by running the main benchmark.

### Committed demo outputs

`results/notebook_demo/` contains the small set of pre-rendered figures and
their input summary CSVs that the notebook displays:

```text
results/notebook_demo/
+-- selected_models_raw_lines.png
+-- selected_models_mask_ratio_summary.csv
+-- method_comparison_boxplots_ratio_60.png
+-- method_comparison_boxplot_summary.csv
+-- 151673_60_high_signal_ge_0.05.png
```

The two summary CSVs are aggregated from per-run `layered_metrics.csv` files
produced by full benchmark runs; the line plot and the boxplot can be
regenerated from them by running `visualization/plot_selected_models_mask_ratio.py`
and `visualization/plot_method_comparison_boxplots.py --mask_ratio 60`. The
3x3 heatmap requires the per-method `pred_ratio.npy` files (not committed; see
the notebook's Section 8 for the rationale) and is regenerated by
`visualization/method_comparison_high_signal_block.py` once those `.npy`
files exist.

## Data

The 12 AnnData `.h5ad` files used by this artifact total roughly 711 MB and
are not bundled in the submitted archive because they exceed the submission
size limit. They are hosted as a shared folder on Google Drive:

<https://drive.google.com/drive/folders/1zBNEj5hOk-h4vSlLV7Ed-rYxFu-d0JRd?usp=drive_link>

Open the link and download the `data` folder (Google Drive will package it
into a `.zip` automatically when downloading a folder). Extract the resulting
archive into the project root so that the final layout is exactly:

```text
data/
  <sample_id>/
    adata_ai_compressed.h5ad
```

Sample IDs included: `151507, 151508, 151509, 151510, 151669, 151670, 151671, 151672, 151673, 151674, 151675, 151676`.

If you already have the `.h5ad` files locally from a previous copy of the
repository, place them at `data/<sample_id>/adata_ai_compressed.h5ad` and skip
the download.

## Quick Start

Run a dry-run first to validate paths and entry points:

```powershell
python experiments\run_harder_benchmark.py --sample_id 151673 --seed 1531 --mask_seed 1531 --model_seed 1531 --epochs 1 --device cpu --models all --dry_run --skip_dual_block
```

Run the default 60% per-site holdout benchmark:

```powershell
python experiments\run_harder_benchmark.py --sample_id 151673 --seed 1531 --mask_seed 1531 --model_seed 1531 --epochs 100 --device cuda --models all --mask_name per_site_random_60_seed{seed} --skip_existing
```

Use `--device cpu` if CUDA is unavailable.

## Main Scripts

| Script | Purpose |
| --- | --- |
| `experiments/run_harder_benchmark.py` | Main benchmark runner. Generates/reuses masks, runs selected models, writes `model_registry.csv`, computes layered metrics, and optionally creates dual-block heatmaps. |
| `src/masks/generate_masks.py` | Shared train/holdout mask generator. |
| `src/models/baselines/train.py` | Runs Mean, SoftImpute, Multiple Imputation, Spatial KNN, and Spatial IDW. |
| `src/models/AIRGate-ST/train.py` | AIRGate-ST training entry point. |
| `src/models/AIRDiff-ST/train.py` | AIRDiff-ST training entry point. |
| `evaluation/collect_layered_metrics.py` | Computes layered metrics and validates NaN/zero semantics. |
| `evaluation/paired_significance.py` | Paired bootstrap summaries for comparing models on the same holdout cells. |
| `evaluation/model_registry_template.csv` | Optional template for manually listing model prediction files when running metric evaluation outside the main benchmark runner. |
| `visualization/plot_mask_ratio_metrics.py` | Optional helper for plotting per-method metric curves across holdout ratios from existing `layered_metrics.csv` files. |

## Expected Outputs

A full benchmark run creates:

```text
results/<run_name>/
+-- model_registry.csv
+-- masks_<sample_id>_seed<seed>/
+-- baselines/
+-- AIRGate-ST/
+-- AIRDiff-ST/
+-- metrics/layered_metrics.csv
+-- dual_block_heatmaps/              # unless --skip_dual_block is used
```

The run-specific `model_registry.csv` is generated automatically by
`experiments/run_harder_benchmark.py`. If predictions are produced manually,
`evaluation/model_registry_template.csv` shows the expected two-column format:
`method,pred_path`.

Important arrays:

```text
pred_ratio.npy       # model prediction matrix
gt_ratio.npy         # ground-truth ratio; NaN outside observed entries
train_ratio.npy      # training-visible ratio; NaN outside train_mask
train_mask.npy       # entries visible for training
val_mask.npy         # held-out entries used for metrics
observed_mask.npy    # train_mask | val_mask
```

## Reproducing Figures

Selected-model mask-ratio curves:

```powershell
python visualization\plot_selected_models_mask_ratio.py
```

Method comparison boxplots across datasets:

```powershell
python visualization\plot_method_comparison_boxplots.py --mask_ratio 60
```

AIRGate-ST cross-dataset mask-ratio summary:

```powershell
python visualization\plot_AIRGate-ST_dataset_mask_ratios.py
```

Optional per-method mask-ratio metric curves:

```powershell
python visualization\plot_mask_ratio_metrics.py
```

Dual-block heatmaps:

```powershell
python visualization\generate_all_dual_blocks.py
```

`visualization/dual_block_observed_heatmap.py` is imported as a module by `generate_all_dual_blocks.py` (it provides the panel-rendering and column-selection helpers) and is not intended to be run directly.

3x3 method-comparison heatmap on a high-signal block. Renders a single 3x3 PNG with `GT`, `Masked Train`, and the seven core methods' predictions on the same observed-only sub-matrix (default mode: `high_signal_ge_0.05`):

```powershell
python visualization\method_comparison_high_signal_block.py `
    --run_dir results\seed1531_e100_per_site_random_60_seed1531 `
    --out_path results\visualization\method_comparison_high_signal_block\151673_60_high_signal_ge_0.05.png
```

The script reads `model_registry.csv` from `--run_dir` and the shared `gt_ratio.npy` / `train_ratio.npy` / `val_mask.npy` from `<run_dir>/<baselines_subdir>/`. The `--baselines_subdir` flag defaults to `baselines` to match the layout produced by `experiments/run_harder_benchmark.py`. Non-core methods that may appear in `model_registry.csv` are silently ignored; the seven core methods are defined by `CORE_METHODS` inside the script.

## Reproducibility

See `REPRODUCIBILITY.md` for the complete reproduction protocol.

Key controls:

- `mask_seed` controls which observed entries are held out.
- `model_seed` controls neural-model initialization and stochastic training.
- `mask_name` controls holdout strategy and ratio.
- `val_mask` entries must have finite predictions for metrics.
- Unobserved entries are not used for metrics.

## Evaluation Summary

Layered metrics include RMSE, MAE, cosine similarity, and Pearson correlation. The two primary reporting layers are:

- `all_holdout`: all held-out observed entries.
- `gt_positive`: held-out observed entries with positive ground-truth signal.

This avoids mixing truly unobserved entries with observed zeros.

## Observations from the Benchmark

The notebook includes a dedicated `Observations from the Benchmark` section for descriptive analysis of the observed A-to-I editing matrix and benchmark figures. It reports three data-level observations:

- Observed A-to-I editing measurements are sparse across the spot-site matrix.
- Positive editing signal is concentrated in a subset of sites rather than uniformly distributed across all sites.
- Spatial spots differ in observed editing support and mean observed editing ratio, indicating spatial heterogeneity in measurable editing signal.

Large per-method prediction matrices are stored as `.npy` during local runs and are not included in the lightweight demo package because of file size. The demo therefore keeps reviewable CSV summaries and PNG figures, while the full `.npy` artifacts remain reproducible by running the documented commands.

## Model Interpretation

AIRGate-ST is interpreted through its two-stage prediction mechanism rather than through a generic tabular SHAP workflow. The model decomposes each prediction into an editability gate and an editing-strength ratio:

```text
prediction = gate_probability * conditional_editing_ratio
```

The gate estimates whether a site is likely to be edited in a spot. This is important because A-to-I editing matrices are highly zero-inflated: an observed zero is biologically different from an unobserved entry. The ratio branch then estimates the expected editing level conditional on the entry being editable.

Site-level bias terms provide an interpretable site prior. Sites that are frequently edited in the training observations receive different initial and learned tendencies from sites that are rarely edited. This helps separate site-specific editing propensity from spot-specific spatial variation.

The spatial component uses neighboring spots as local support. AIRGate-ST aggregates spatial KNN information and uses a learned attention gate to adjust editability based on nearby observed context. In practical terms, predictions for spatially coherent editing patterns should be supported by neighboring spots, while training-observed values are preserved exactly in the final prediction matrix.

This mechanism-level interpretation is reported together with layered metrics: `all_holdout` evaluates all held-out observed entries, while `gt_positive` focuses on held-out entries with positive editing signal.

## Attribution and AI Use

- Dependency and originality notes: `ATTRIBUTION.md`
- AI use declaration: `AI_USE_DECLARATION.md`
