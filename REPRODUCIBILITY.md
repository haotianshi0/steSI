# Reproducibility Protocol

This document gives the exact sequence needed to reproduce the core benchmark outputs for `Spatial transcriptome and epitranscriptome: spatial imputation` from a clean copy of the `steSI` artifact folder.

The workflow follows a typical spatial transcriptomics model-artifact structure: environment setup, data placement, smoke test, benchmark execution, metric collection, and figure generation.

## 1. Prepare Environment

In the commands below, `<PROJECT_ROOT>` means the local folder that contains `README.md`, `requirements.txt`, `environment.yml`, `experiments/`, `src/`, and `visualization/`.

```powershell
cd <PROJECT_ROOT>
conda env create -f environment.yml
conda activate st_ai_impute
```

If the environment already exists:

```powershell
conda activate st_ai_impute
python -m pip install -r requirements.txt
```

## 2. Confirm Data Layout

Each dataset must be stored as:

```text
data/<sample_id>/adata_ai_compressed.h5ad
```

The artifact was prepared with sample IDs:

```text
151507, 151508, 151509, 151510,
151669, 151670, 151671, 151672,
151673, 151674, 151675, 151676
```

## 3. Smoke Test

Run a dry-run first. This does not train models, but confirms that all paths and entry points resolve correctly:

```powershell
python experiments\run_harder_benchmark.py --sample_id 151673 --seed 1531 --mask_seed 1531 --model_seed 1531 --epochs 1 --device cpu --models all --dry_run --skip_dual_block
```

Expected behavior: the command prints the planned calls to mask generation, baselines, AIRGate-ST, AIRDiff-ST, and layered metrics without raising errors.

This step is recommended before any long GPU run because it catches path, environment, and script-entry errors without generating large outputs.

## 4. Reproduce Main Benchmark

```powershell
python experiments\run_harder_benchmark.py --sample_id 151673 --seed 1531 --mask_seed 1531 --model_seed 1531 --epochs 100 --device cuda --models all --mask_name per_site_random_60_seed{seed} --skip_existing
```

Use `--device cpu` if CUDA is unavailable.

Expected output:

```text
results/seed1531_e100_per_site_random_60_seed1531/
```

The key file for reporting is:

```text
results/seed1531_e100_per_site_random_60_seed1531/metrics/layered_metrics.csv
```

The benchmark runner also writes a run-specific `model_registry.csv`, which maps each method name to its `pred_ratio.npy` file. For manual evaluation outside the runner, `evaluation/model_registry_template.csv` provides the expected `method,pred_path` format.

## 5. Reproduce Mask-Ratio Results

The reported mask-ratio experiments use 20%, 40%, and 60% holdout.

```powershell
python experiments\run_151673_mask_ratios.py
```

## 6. Reproduce Cross-Dataset AIRGate-ST Supplement

```powershell
python experiments\run_AIRGate-ST_mask_ratios_other_datasets.py
```

## 7. Reproduce Figures

```powershell
python visualization\plot_selected_models_mask_ratio.py
python visualization\plot_method_comparison_boxplots.py --mask_ratio 60
python visualization\plot_AIRGate-ST_dataset_mask_ratios.py
python visualization\plot_mask_ratio_metrics.py
python visualization\generate_all_dual_blocks.py
python visualization\method_comparison_high_signal_block.py `
    --run_dir results\seed1531_e100_per_site_random_60_seed1531 `
    --out_path results\visualization\method_comparison_high_signal_block\151673_60_high_signal_ge_0.05.png
```

`method_comparison_high_signal_block.py` produces a single 3x3 PNG with `GT`, `Masked Train`, and the seven core methods' predictions on a shared high-signal observed-only sub-matrix  (defaults: `--high_signal_threshold 0.05`, 32-64 spots, 50 sites). It reads `model_registry.csv` and the `gt_ratio.npy` / `train_ratio.npy` / `val_mask.npy` written under `--baselines_subdir` (default: `baselines`).

`visualization/plot_mask_ratio_metrics.py` is an optional helper that plots per-method metric curves across holdout ratios from existing `layered_metrics.csv` files.

Figure outputs are written under:

```text
results/visualization/
```

or the run-specific result folders, depending on the script.

A small set of pre-rendered demo outputs is committed under `results/notebook_demo/` so that `artifact_walkthrough.ipynb` can display benchmark figures without first requiring a local benchmark run:

```text
results/notebook_demo/selected_models_raw_lines.png
results/notebook_demo/selected_models_mask_ratio_summary.csv
results/notebook_demo/method_comparison_boxplots_ratio_60.png
results/notebook_demo/method_comparison_boxplot_summary.csv
results/notebook_demo/151673_60_high_signal_ge_0.05.png
```

The two summary CSVs are long-form aggregations of the per-run `layered_metrics.csv` files that the boxplot and line-plot scripts read. Per-method `pred_ratio.npy` files are not committed (each one is roughly 60 MB; see Section 8 of `artifact_walkthrough.ipynb`); the high-signal heatmap PNG is therefore the only `notebook_demo` artifact that requires a full benchmark run to regenerate.

## 8. Review Notebook Observations

Open the executable notebook from the project root:

```powershell
jupyter lab artifact_walkthrough.ipynb
```

Run through the `Observations from the Benchmark` section to review the descriptive observations and the pre-rendered demo figures. The notebook is intended as a readable artifact walkthrough; it does not create additional result files by default. Full prediction matrices are generated as `.npy` files during local benchmark runs, but are not included in the lightweight demo package because of file size.

## 9. Reproduce Significance Summaries

After multi-seed runs have been produced, run:

```powershell
python experiments\run_seed_ensemble.py --sample_id 151673 --mask_seed 10 --model_seeds 6,10,42 --epochs 100 --device cuda --shared_mask_cache --skip_existing
```

This creates aggregate model-seed metrics, prediction-level ensembles, and paired significance summaries.

Note: `run_seed_ensemble.py` automatically invokes `experiments/build_prediction_ensemble.py` as a subprocess to produce the per-model prediction-level ensembles. There is no need to run `build_prediction_ensemble.py` manually; it is listed only because it appears in the `experiments/` directory.

## 10. Reproducibility Notes

- `mask_seed` controls which observed entries become holdout entries.
- `model_seed` controls model initialization and stochastic training.
- `train_mask | val_mask == observed_mask` is required and validated.
- `val_mask` entries must have finite predictions for metrics.
- Unobserved entries are not used for metrics.
- NaN and zero have different meanings and must not be collapsed.

## 11. Result File Checklist

After a successful full run, confirm that these files exist:

```text
results/<run_name>/model_registry.csv
results/<run_name>/metrics/layered_metrics.csv
results/<run_name>/masks_<sample_id>_seed<seed>/<mask_name>/train_mask.npy
results/<run_name>/masks_<sample_id>_seed<seed>/<mask_name>/val_mask.npy
results/<run_name>/masks_<sample_id>_seed<seed>/<mask_name>/observed_mask.npy
results/<run_name>/AIRGate-ST/pred_ratio.npy
results/<run_name>/AIRDiff-ST/pred_ratio.npy
results/<run_name>/baselines/mean_baseline_ratio.npy
```
