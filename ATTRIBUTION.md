# Attribution and Original Work

## Original Project Components

The following components were developed for this project:

- AIRGate-ST: a two-stage spatial editing-ratio model with a gate component and ratio component for A-to-I editing imputation.
- AIRDiff-ST: a conditional diffusion-style model for held-out editing-ratio imputation. Its training entry point reuses the project's shared data-loading and metric utilities (`src/models/AIRGate-ST/shared/data_utils.py`) instead of redefining them; the model architecture, conditioning, noise schedule, and sampling loop are original to AIRDiff-ST.
- Shared train/holdout mask generation with explicit NaN/zero semantics.
- Layered metric evaluation for all holdout and positive-signal subsets.
- Dual-block heatmap and mask-ratio visualization scripts.
- Benchmark orchestration across baselines, neural models, datasets, and mask ratios.

## Open-Source Dependencies

This artifact builds on established open-source libraries:

- NumPy and SciPy for numerical computing.
- pandas for tabular result processing.
- scikit-learn for SoftImpute-related randomized SVD components and iterative imputation utilities.
- PyTorch for neural model training.
- AnnData and h5py for `.h5ad` data loading.
- Matplotlib for visualization.

## Naming Notes

The formal model names used in the final artifact are:

```text
AIRGate-ST
AIRDiff-ST
```