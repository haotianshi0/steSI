"""Shared utilities for mask generation and the official baseline models.

This module intentionally keeps only the functions used by the final artifact
production workflow:

- `src/masks/generate_masks.py`
- `src/models/baselines/train.py`

Historical exploratory baselines such as NMF, graph Laplacian smoothing,
autoencoders, and GNN prototypes were removed from this artifact-facing helper
to keep the dependency surface small and avoid exposing unused model code.
"""

import os
from typing import Dict, Tuple

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


def to_dense_float(matrix) -> np.ndarray:
    """Return a dense float32 array from either sparse or dense input."""
    if sparse.issparse(matrix):
        return matrix.toarray().astype(np.float32, copy=False)
    return np.asarray(matrix, dtype=np.float32)


def positive_site_counts(g_counts) -> np.ndarray:
    """Count positive G observations per editing site."""
    if sparse.issparse(g_counts):
        return np.asarray(g_counts.getnnz(axis=0)).ravel()
    return np.asarray((g_counts > 0).sum(axis=0)).ravel()


def choose_site_mask(g_counts, initial_threshold: int) -> Tuple[np.ndarray, int]:
    """Choose sites with enough positive G support, relaxing the threshold if needed."""
    counts = positive_site_counts(g_counts)
    for threshold in [initial_threshold, 8, 5, 2, 0]:
        mask = counts > threshold
        if int(mask.sum()) > 0:
            return mask, threshold
    return counts > -1, -1


def compute_ratio(a_counts: np.ndarray, g_counts: np.ndarray, missing_as_nan: bool) -> np.ndarray:
    """Compute G / (A + G), optionally marking invalid entries as NaN."""
    denom = a_counts + g_counts
    valid = np.isfinite(denom) & (denom > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(
            g_counts,
            denom,
            out=np.zeros_like(g_counts, dtype=np.float32),
            where=valid,
        )
    if missing_as_nan:
        ratio = ratio.astype(np.float32, copy=False)
        ratio[~valid] = np.nan
    return ratio.astype(np.float32, copy=False)


def apply_min_depth_filter(
    a_counts: np.ndarray,
    g_counts: np.ndarray,
    min_depth: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Mask entries whose A+G depth is below the required observation threshold."""
    a_filtered = a_counts.copy()
    g_filtered = g_counts.copy()

    raw_observed_mask = (a_counts + g_counts) > 0
    raw_valid = int(raw_observed_mask.sum())
    total_entries = int(a_counts.size)
    raw_missing_rate = 1.0 - (raw_valid / total_entries)

    high_conf_mask = (a_counts + g_counts) >= min_depth
    a_filtered[~high_conf_mask] = np.nan
    g_filtered[~high_conf_mask] = np.nan

    filtered_observed_mask = ~np.isnan(a_filtered)
    filtered_valid = int(filtered_observed_mask.sum())
    filtered_missing_rate = 1.0 - (filtered_valid / total_entries)

    stats = {
        "raw_valid_observations": raw_valid,
        "raw_missing_rate": float(raw_missing_rate),
        "filtered_valid_observations": filtered_valid,
        "filtered_missing_rate": float(filtered_missing_rate),
    }
    return a_filtered, g_filtered, filtered_observed_mask, stats


def log1p_transform(matrix: np.ndarray) -> np.ndarray:
    """Apply a non-negative log1p transform to count-like values."""
    return np.log1p(np.clip(matrix, a_min=0.0, a_max=None)).astype(np.float32)


def expm1_inverse(matrix: np.ndarray) -> np.ndarray:
    """Invert `log1p_transform` while keeping outputs non-negative."""
    return np.expm1(np.clip(matrix, a_min=0.0, a_max=None)).astype(np.float32)


def protect_all_nan_columns(train_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Replace all-NaN columns with zeros so downstream imputers stay finite."""
    all_nan_cols = np.all(np.isnan(train_matrix), axis=0)
    protected = train_matrix.copy()
    if np.any(all_nan_cols):
        protected[:, all_nan_cols] = 0.0
    return protected.astype(np.float32), all_nan_cols


def mean_impute_channel(train_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Column-mean imputation used internally as fallback support."""
    imputed, _ = protect_all_nan_columns(train_matrix)
    col_means = np.nanmean(imputed, axis=0)
    global_mean = float(np.nanmean(imputed))
    col_means = np.where(np.isnan(col_means), global_mean, col_means).astype(np.float32)
    missing = np.isnan(imputed)
    imputed[missing] = np.take(col_means, np.where(missing)[1])
    return imputed.astype(np.float32), col_means


def zero_inflation_gate(neighbor_vals: np.ndarray, positive_threshold: float) -> np.ndarray:
    """Estimate local positive-signal support from neighboring values."""
    return np.mean(neighbor_vals > positive_threshold, axis=0).astype(np.float32)


def resolve_n_jobs(n_jobs: int | None) -> int:
    """Resolve job count semantics shared by Spatial KNN and Spatial IDW."""
    if n_jobs is None or n_jobs == 0:
        return max(1, os.cpu_count() or 1)
    if n_jobs < 0:
        return max(1, (os.cpu_count() or 1) + 1 + n_jobs)
    return max(1, n_jobs)


def spatial_knn_impute_channel(
    train_matrix: np.ndarray,
    coords: np.ndarray,
    n_neighbors: int,
    positive_threshold: float = 0.1,
    gate_threshold: float = 0.15,
    n_jobs: int = 1,
) -> np.ndarray:
    """Spatial K-nearest-neighbor imputation with uniform neighbor weights.

    This is intentionally distinct from `idw_impute_channel`, which uses
    inverse-distance-squared weights. The queried spot itself is excluded from
    the neighbor set.
    """
    train_matrix, all_nan_cols = protect_all_nan_columns(train_matrix)
    n_jobs = resolve_n_jobs(n_jobs)
    n_obs = train_matrix.shape[0]
    k = min(max(2, n_neighbors), n_obs - 1)
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=n_jobs)
    nbrs.fit(coords)
    _, indices = nbrs.kneighbors(coords)

    imputed = train_matrix.copy()
    _, col_means = mean_impute_channel(train_matrix)

    def compute_row(i: int):
        missing_cols = np.isnan(imputed[i])
        if not np.any(missing_cols):
            return None
        neigh_idx = indices[i, 1:]
        neigh_vals = train_matrix[neigh_idx]
        valid = ~np.isnan(neigh_vals)
        weighted_sum = np.where(valid, neigh_vals, 0.0).sum(axis=0)
        weight_sum = valid.sum(axis=0).astype(np.float32)
        pred = np.divide(
            weighted_sum,
            weight_sum,
            out=np.full(train_matrix.shape[1], np.nan, dtype=np.float32),
            where=weight_sum > 0,
        )
        positive_frac = zero_inflation_gate(neigh_vals, positive_threshold)
        pred = np.where(positive_frac < gate_threshold, 0.0, pred)
        pred = np.where(np.isnan(pred), col_means, pred)
        return i, missing_cols, pred[missing_cols]

    if n_jobs == 1:
        rows = (compute_row(i) for i in range(n_obs))
    else:
        from joblib import Parallel, delayed

        rows = Parallel(n_jobs=n_jobs, prefer="threads", batch_size=16)(
            delayed(compute_row)(i) for i in range(n_obs)
        )
    for row in rows:
        if row is None:
            continue
        i, missing_cols, values = row
        imputed[i, missing_cols] = values
    if np.any(all_nan_cols):
        imputed[:, all_nan_cols] = 0.0
    return np.clip(imputed, 0.0, None).astype(np.float32)


def idw_impute_channel(
    train_matrix: np.ndarray,
    coords: np.ndarray,
    n_neighbors: int,
    power: float = 2.0,
    positive_threshold: float = 0.1,
    gate_threshold: float = 0.15,
    n_jobs: int = 1,
) -> np.ndarray:
    """Spatial inverse-distance weighted imputation for one count channel."""
    train_matrix, all_nan_cols = protect_all_nan_columns(train_matrix)
    n_jobs = resolve_n_jobs(n_jobs)
    n_obs = train_matrix.shape[0]
    k = min(max(2, n_neighbors), n_obs)
    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=n_jobs)
    nbrs.fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    imputed = train_matrix.copy()
    _, col_means = mean_impute_channel(train_matrix)

    def compute_row(i: int):
        missing_cols = np.isnan(imputed[i])
        if not np.any(missing_cols):
            return None
        neigh_idx = indices[i, 1:]
        neigh_dist = distances[i, 1:]
        neigh_vals = train_matrix[neigh_idx]
        weights = 1.0 / np.power(neigh_dist + 1e-6, power)
        weights = weights[:, None]
        valid = ~np.isnan(neigh_vals)
        weighted_sum = np.where(valid, neigh_vals * weights, 0.0).sum(axis=0)
        weight_sum = np.where(valid, weights, 0.0).sum(axis=0)
        pred = np.divide(
            weighted_sum,
            weight_sum,
            out=np.full(train_matrix.shape[1], np.nan, dtype=np.float32),
            where=weight_sum > 0,
        )
        positive_frac = zero_inflation_gate(neigh_vals, positive_threshold)
        pred = np.where(positive_frac < gate_threshold, 0.0, pred)
        pred = np.where(np.isnan(pred), col_means, pred)
        return i, missing_cols, pred[missing_cols]

    if n_jobs == 1:
        rows = (compute_row(i) for i in range(n_obs))
    else:
        from joblib import Parallel, delayed

        rows = Parallel(n_jobs=n_jobs, prefer="threads", batch_size=16)(
            delayed(compute_row)(i) for i in range(n_obs)
        )
    for row in rows:
        if row is None:
            continue
        i, missing_cols, values = row
        imputed[i, missing_cols] = values
    if np.any(all_nan_cols):
        imputed[:, all_nan_cols] = 0.0
    return np.clip(imputed, 0.0, None).astype(np.float32)


def run_separate_channel_imputer(
    train_a_log: np.ndarray,
    train_g_log: np.ndarray,
    imputer_fn,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the same single-channel imputer to A and G channels separately."""
    a_out = imputer_fn(train_a_log, **kwargs)
    g_out = imputer_fn(train_g_log, **kwargs)
    if isinstance(a_out, tuple):
        a_out = a_out[0]
    if isinstance(g_out, tuple):
        g_out = g_out[0]
    return a_out.astype(np.float32), g_out.astype(np.float32)


def reconstruct_ratio_from_log_counts(
    imputed_a_log: np.ndarray,
    imputed_g_log: np.ndarray,
    observed_a_counts: np.ndarray,
    observed_g_counts: np.ndarray,
    train_observed_mask: np.ndarray,
    test_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct full and evaluation-only editing ratios from imputed log counts."""
    imputed_a_counts = expm1_inverse(imputed_a_log)
    imputed_g_counts = expm1_inverse(imputed_g_log)
    # Only restore original counts where they are actually known. Restoring
    # NaN-ed holdout positions would erase the prediction being evaluated.
    safe_mask = (
        train_observed_mask
        & ~np.isnan(observed_a_counts)
        & ~np.isnan(observed_g_counts)
    )
    imputed_a_counts[safe_mask] = observed_a_counts[safe_mask]
    imputed_g_counts[safe_mask] = observed_g_counts[safe_mask]
    ratio_full = compute_ratio(imputed_a_counts, imputed_g_counts, missing_as_nan=False)
    return ratio_full.astype(np.float32), ratio_full[test_mask].astype(np.float32)
