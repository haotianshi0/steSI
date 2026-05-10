"""Train and save the five non-neural baseline imputers.

The output prediction matrices preserve all training-visible ground-truth values
exactly and are validated before being written to disk.
"""

import argparse
import json
import os
import sys

import anndata as ad
import numpy as np


DEFAULT_SOURCE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def add_source_paths(source_root: str) -> None:
    """Add the project's helper directories to ``sys.path``.

    Required so the deferred imports of ``shared.data_utils`` and
    ``baseline_utils`` inside :func:`main` resolve regardless of the directory
    the script is invoked from.
    """
    sys.path.insert(0, os.path.join(source_root, "src", "models", "AIRGate-ST"))
    sys.path.insert(0, os.path.join(source_root, "src", "shared"))


def softimpute_ratio(gt_ratio: np.ndarray, train_mask: np.ndarray,
                     max_iters: int = 100, convergence_threshold: float = 0.001,
                     max_rank: int = 100, shrinkage: float | None = None,
                     random_state: int = 0) -> np.ndarray:
    """In-house SoftImpute (Mazumder/Hastie/Tibshirani 2010): iterative truncated SVD with
    soft-thresholded singular values, observed entries kept fixed.

    Replaces fancyimpute.SoftImpute (which is no longer compatible with sklearn>=1.8 because of
    the removed `force_all_finite` kwarg). Uses sklearn.randomized_svd to keep cost tractable on
    the 3639x4698 ratio matrix; max_rank caps the truncated SVD rank, shrinkage defaults to
    sigma_max / 50 if not provided (matches fancyimpute's default heuristic).
    """
    from sklearn.utils.extmath import randomized_svd

    obs = train_mask.astype(bool)
    M = gt_ratio.astype(np.float64)
    if not obs.any():
        return np.zeros_like(M, dtype=np.float32)

    col_count = obs.sum(axis=0).astype(np.float64)
    col_sum = np.where(obs, M, 0.0).sum(axis=0)
    col_mean = np.where(col_count > 0, col_sum / np.maximum(col_count, 1.0), 0.0)
    X = np.where(obs, M, col_mean[np.newaxis, :])

    rank_cap = max(1, min(max_rank, min(X.shape) - 1))
    if shrinkage is None:
        _, S0, _ = randomized_svd(X, n_components=1, random_state=random_state)
        shrinkage = float(S0[0]) / 50.0

    prev = X.copy()
    prev_norm = max(np.linalg.norm(prev), 1e-9)
    for _ in range(max_iters):
        U, S, Vt = randomized_svd(X, n_components=rank_cap, random_state=random_state)
        S_thresh = np.maximum(S - shrinkage, 0.0)
        keep = int(np.sum(S_thresh > 0))
        if keep == 0:
            X_low = np.zeros_like(X)
        else:
            X_low = (U[:, :keep] * S_thresh[:keep]) @ Vt[:keep, :]
        X = np.where(obs, M, X_low)
        delta = np.linalg.norm(X - prev) / prev_norm
        if delta < convergence_threshold:
            break
        prev = X.copy()
        prev_norm = max(np.linalg.norm(prev), 1e-9)

    pred = np.clip(X, 0.0, 1.0).astype(np.float32)
    pred[obs] = gt_ratio[obs].astype(np.float32)
    return pred


def multiple_imputation_ratio(gt_ratio: np.ndarray, train_mask: np.ndarray,
                              m: int = 3, max_iter: int = 5,
                              n_nearest_features: int = 50,
                              random_state: int = 0) -> np.ndarray:
    """Multiple-imputation baseline via sklearn IterativeImputer with sample_posterior=True.

    Produces m completed datasets by drawing from the posterior of a Bayesian-ridge regression
    fitted column-wise (chained equations / MICE-style). The point-estimate output here is the
    arithmetic mean across the m draws (Rubin's pooling for the mean estimand on a per-cell
    basis). Note this is a pragmatic MI for very wide matrices: n_nearest_features=50 limits each
    column's regression to its 50 most-correlated peers, which is the sklearn-recommended way to
    keep MICE tractable on >>1000 columns. It is *not* a textbook full-conditional MICE.
    """
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import BayesianRidge

    train_ratio = np.where(train_mask, gt_ratio, np.nan).astype(np.float64)
    accum = np.zeros_like(train_ratio)
    for i in range(m):
        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            sample_posterior=True,
            max_iter=max_iter,
            n_nearest_features=n_nearest_features,
            initial_strategy="mean",
            min_value=0.0,
            max_value=1.0,
            keep_empty_features=True,
            random_state=random_state + i,
        )
        accum += imputer.fit_transform(train_ratio)
    pred = (accum / float(m)).astype(np.float32)
    pred = np.clip(pred, 0.0, 1.0)
    pred[train_mask] = gt_ratio[train_mask].astype(np.float32)
    return pred


def mean_baseline(gt_ratio: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    """Per-site mean imputation.

    For each site (column) the prediction is the mean of all training-visible
    ratios at that site. Sites with no training observations fall back to the
    global training mean. Training-visible cells are written through with
    their exact ground-truth ratio.
    """
    train_values = np.where(train_mask, gt_ratio, np.nan)
    col_mean = np.nanmean(train_values, axis=0)
    global_mean = float(np.nanmean(train_values))
    col_mean = np.where(np.isnan(col_mean), global_mean, col_mean)
    pred = np.broadcast_to(col_mean.reshape(1, -1), gt_ratio.shape).copy().astype(np.float32)
    pred[train_mask] = gt_ratio[train_mask]
    return pred


def validate_prediction_output(name: str, pred: np.ndarray, gt_ratio: np.ndarray,
                               train_mask: np.ndarray, val_mask: np.ndarray) -> np.ndarray:
    """Verify a baseline prediction matrix before it is written to disk.

    Checks: shape match with ``gt_ratio``; finiteness on observed (train +
    val) entries; finite values constrained to ``[0, 1]``; ``val_mask``
    entries finite (so metric scoring can proceed); training-visible entries
    exactly equal to ``gt_ratio``. Raises ``ValueError`` on any violation.
    """
    pred = pred.astype(np.float32, copy=False)
    if pred.shape != gt_ratio.shape:
        raise ValueError(f"{name}: shape mismatch pred={pred.shape}, gt={gt_ratio.shape}")
    pred_finite = np.isfinite(pred)
    observed_mask = train_mask | val_mask
    if np.any(observed_mask & ~pred_finite):
        raise ValueError(f"{name}: observed train/holdout entries contain NaN/Inf")
    finite_pred = pred[pred_finite]
    if finite_pred.size and np.any((finite_pred < 0.0) | (finite_pred > 1.0)):
        raise ValueError(f"{name}: finite prediction outside [0, 1]")
    if np.any(val_mask & ~np.isfinite(pred)):
        raise ValueError(f"{name}: metric/holdout entries contain NaN")
    if not np.allclose(pred[train_mask], gt_ratio[train_mask], equal_nan=False):
        raise ValueError(f"{name}: train entries must preserve all training-visible gt_ratio values exactly")
    return pred


def main() -> None:
    """CLI entry point: load the h5ad, apply the external masks, and write the
    five baseline prediction files plus the matching ground-truth and mask
    arrays into ``--out_dir``.

    Steps:

      1. Load the AnnData h5ad and apply the same site-frequency and
         min-depth filters used by the neural training scripts.
      2. Load the external ``train_mask.npy`` / ``val_mask.npy`` from
         ``--mask_dir`` and validate that ``train_mask | val_mask`` exactly
         equals the resulting ``observed_mask``.
      3. Compute predictions for the five baselines (Mean, SoftImpute,
         Multiple Imputation, Spatial KNN, Spatial IDW). Each is validated
         and written as ``<method>_ratio.npy``. Spatial KNN/IDW operate on
         log-transformed A/G counts and reconstruct the ratio from the
         predicted log-counts.
      4. Copy mask and ground-truth arrays into ``--out_dir`` so downstream
         visualization scripts can read them from a single directory.
      5. Write a small JSON summary with applied thresholds and depth stats.
    """
    ap = argparse.ArgumentParser(
        description="Run baseline models: Mean, SoftImpute, Multiple Imputation, Spatial KNN, and Spatial IDW."
    )
    ap.add_argument("--source_root", default=DEFAULT_SOURCE_ROOT)
    ap.add_argument("--h5ad", default=os.path.join(DEFAULT_SOURCE_ROOT, "data", "151673", "adata_ai_compressed.h5ad"))
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--threshold", type=int, default=10)
    ap.add_argument("--min_depth", type=int, default=10)
    ap.add_argument("--spatial_knn_k", type=int, default=12)
    ap.add_argument("--idw_k", type=int, default=12)
    ap.add_argument("--n_jobs", type=int, default=0, help="CPU workers for spatial KNN/IDW. 0 means all logical cores.")
    ap.add_argument("--softimpute_max_iters", type=int, default=100)
    ap.add_argument("--softimpute_tol", type=float, default=0.001)
    ap.add_argument("--softimpute_max_rank", type=int, default=100)
    args = ap.parse_args()

    add_source_paths(args.source_root)
    from shared.data_utils import compute_ratio  # noqa: WPS433
    from baseline_utils import (  # noqa: WPS433
        apply_min_depth_filter,
        choose_site_mask,
        idw_impute_channel,
        log1p_transform,
        reconstruct_ratio_from_log_counts,
        run_separate_channel_imputer,
        spatial_knn_impute_channel,
        to_dense_float,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    adata = ad.read_h5ad(args.h5ad)
    site_mask, used_threshold = choose_site_mask(adata.layers["G"], args.threshold)
    adata_f = adata[:, site_mask].copy()
    raw_a = to_dense_float(adata_f.layers["A"])
    raw_g = to_dense_float(adata_f.layers["G"])
    a_counts, g_counts, observed_mask, depth_stats = apply_min_depth_filter(raw_a, raw_g, args.min_depth)

    train_mask = np.load(os.path.join(args.mask_dir, "train_mask.npy")).astype(bool)
    val_mask = np.load(os.path.join(args.mask_dir, "val_mask.npy")).astype(bool)
    if train_mask.shape != a_counts.shape or val_mask.shape != a_counts.shape:
        raise ValueError(f"Mask shape mismatch: train={train_mask.shape}, counts={a_counts.shape}")
    if np.any(train_mask & val_mask):
        raise ValueError("External train_mask and val_mask overlap.")
    if not np.array_equal(train_mask | val_mask, observed_mask):
        missing = int((observed_mask & ~(train_mask | val_mask)).sum())
        outside = int(((train_mask | val_mask) & ~observed_mask).sum())
        raise ValueError(
            f"External masks must satisfy train_mask | val_mask == observed_mask; "
            f"missing_observed={missing}, outside_observed={outside}"
        )

    train_a_counts = np.where(train_mask, a_counts, np.nan).astype(np.float32)
    train_g_counts = np.where(train_mask, g_counts, np.nan).astype(np.float32)
    gt_ratio = compute_ratio(
        np.nan_to_num(a_counts, nan=0.0),
        np.nan_to_num(g_counts, nan=0.0),
        mask=observed_mask,
    ).astype(np.float32)

    mean_pred = validate_prediction_output("mean_baseline", mean_baseline(gt_ratio, train_mask),
                                           gt_ratio, train_mask, val_mask)
    softimpute_pred = validate_prediction_output(
        "softimpute",
        softimpute_ratio(gt_ratio, train_mask,
                         max_iters=args.softimpute_max_iters,
                         convergence_threshold=args.softimpute_tol,
                         max_rank=args.softimpute_max_rank),
        gt_ratio,
        train_mask,
        val_mask,
    )
    mi_pred = validate_prediction_output("multiple_imputation",
                                         multiple_imputation_ratio(gt_ratio, train_mask),
                                         gt_ratio, train_mask, val_mask)

    np.save(os.path.join(args.out_dir, "mean_baseline_ratio.npy"), mean_pred)
    np.save(os.path.join(args.out_dir, "softimpute_ratio.npy"), softimpute_pred)
    np.save(os.path.join(args.out_dir, "mi_ratio.npy"), mi_pred)

    train_a_log = log1p_transform(np.nan_to_num(train_a_counts, nan=0.0))
    train_g_log = log1p_transform(np.nan_to_num(train_g_counts, nan=0.0))
    train_a_log[np.isnan(train_a_counts)] = np.nan
    train_g_log[np.isnan(train_g_counts)] = np.nan

    if "x_pixel" in adata_f.obs.columns and "y_pixel" in adata_f.obs.columns:
        coords = adata_f.obs[["x_pixel", "y_pixel"]].to_numpy(dtype=np.float32)
    elif "spatial" in adata_f.obsm:
        coords = np.asarray(adata_f.obsm["spatial"], dtype=np.float32)
    else:
        raise KeyError("No spatial coordinates found.")

    train_observed_mask = train_mask
    for method_name, imputer_fn, kwargs in [
        ("spatial_knn", spatial_knn_impute_channel, {"coords": coords, "n_neighbors": args.spatial_knn_k, "n_jobs": args.n_jobs}),
        ("spatial_idw", idw_impute_channel, {"coords": coords, "n_neighbors": args.idw_k, "n_jobs": args.n_jobs}),
    ]:
        a_log, g_log = run_separate_channel_imputer(train_a_log, train_g_log, imputer_fn, **kwargs)
        ratio_full, _ = reconstruct_ratio_from_log_counts(
            a_log,
            g_log,
            train_a_counts,
            train_g_counts,
            train_observed_mask,
            val_mask,
        )
        ratio_full = validate_prediction_output(method_name, ratio_full.astype(np.float32),
                                                gt_ratio, train_mask, val_mask)
        np.save(os.path.join(args.out_dir, f"{method_name}_ratio.npy"), ratio_full)

    for name in ["train_mask.npy", "val_mask.npy", "observed_mask.npy", "gt_ratio.npy", "train_ratio.npy"]:
        src = os.path.join(args.mask_dir, name)
        if os.path.exists(src):
            arr = np.load(src)
            np.save(os.path.join(args.out_dir, name), arr)

    summary = {
        "h5ad": os.path.abspath(args.h5ad),
        "mask_dir": os.path.abspath(args.mask_dir),
        "out_dir": os.path.abspath(args.out_dir),
        "used_site_threshold": int(used_threshold),
        "min_depth": int(args.min_depth),
        "n_jobs": int(args.n_jobs),
        "depth_stats": depth_stats,
        "saved_methods": ["mean_baseline", "softimpute", "multiple_imputation",
                          "spatial_knn", "spatial_idw"],
        "unobserved_prediction_count": int((~observed_mask).sum()),
    }
    with open(os.path.join(args.out_dir, "spatial_baseline_external_mask_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
