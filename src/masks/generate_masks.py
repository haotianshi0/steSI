"""Generate reproducible train/holdout masks for observed editing entries.

For every mask case, the saved arrays satisfy:
train_mask and val_mask do not overlap, and train_mask | val_mask equals the
observed_mask derived from depth-filtered A/G counts.
"""

import argparse
import csv
import json
import os
import sys
from typing import Dict, Iterable, Tuple

import anndata as ad
import numpy as np


DEFAULT_SOURCE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def add_source_paths(source_root: str) -> None:
    """Add the project's helper directories to ``sys.path`` for deferred imports."""
    sys.path.insert(0, os.path.join(source_root, "src", "models", "AIRGate-ST"))
    sys.path.insert(0, os.path.join(source_root, "src", "shared"))


def load_filtered_data(source_root: str, h5ad: str, threshold: int, min_depth: int):
    """Load the h5ad and apply the project's site-frequency and min-depth filters.

    Returns a dict with the filtered count matrices, ``observed_mask``,
    ground-truth ratio, spatial coordinates (or ``None``), and bookkeeping
    fields used by ``save_mask_case`` to write a per-mask summary JSON.
    """
    add_source_paths(source_root)
    from shared.data_utils import compute_ratio  # noqa: WPS433
    from baseline_utils import (  # noqa: WPS433
        apply_min_depth_filter,
        choose_site_mask,
        to_dense_float,
    )

    adata = ad.read_h5ad(h5ad)
    site_mask, used_threshold = choose_site_mask(adata.layers["G"], threshold)
    adata_f = adata[:, site_mask].copy()
    raw_a = to_dense_float(adata_f.layers["A"])
    raw_g = to_dense_float(adata_f.layers["G"])
    a_counts, g_counts, observed_mask, depth_stats = apply_min_depth_filter(raw_a, raw_g, min_depth)
    gt_ratio = compute_ratio(
        np.nan_to_num(a_counts, nan=0.0),
        np.nan_to_num(g_counts, nan=0.0),
        mask=observed_mask,
    ).astype(np.float32)

    if "x_pixel" in adata_f.obs.columns and "y_pixel" in adata_f.obs.columns:
        coords = adata_f.obs[["x_pixel", "y_pixel"]].to_numpy(dtype=np.float32)
    elif "spatial" in adata_f.obsm:
        coords = np.asarray(adata_f.obsm["spatial"], dtype=np.float32)
    else:
        coords = None

    return {
        "A": a_counts,
        "G": g_counts,
        "gt_ratio": gt_ratio,
        "observed_mask": observed_mask,
        "coords": coords,
        "site_mask": site_mask,
        "used_threshold": used_threshold,
        "depth_stats": depth_stats,
        "shape": gt_ratio.shape,
    }


def random_entry_mask(observed: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Pick ``frac`` of all observed entries uniformly at random as the val mask."""
    rng = np.random.default_rng(seed)
    idx = np.argwhere(observed)
    n = int(idx.shape[0] * frac)
    val = np.zeros_like(observed, dtype=bool)
    if n > 0:
        picked = rng.choice(idx.shape[0], size=n, replace=False)
        chosen = idx[picked]
        val[chosen[:, 0], chosen[:, 1]] = True
    return val


def per_site_mask(observed: np.ndarray, frac: float, seed: int, min_train_per_site: int) -> np.ndarray:
    """Per-site stratified random mask.

    Within each site (column), independently picks ``frac`` of its observed
    entries for the val mask, while leaving at least ``min_train_per_site``
    observed entries visible to training. This is the strategy used by the
    project's main reporting masks (``per_site_random_{20,40,60}``).
    """
    rng = np.random.default_rng(seed)
    val = np.zeros_like(observed, dtype=bool)
    for site in range(observed.shape[1]):
        rows = np.where(observed[:, site])[0]
        if rows.size <= min_train_per_site:
            continue
        n = min(int(rows.size * frac), rows.size - min_train_per_site)
        if n > 0:
            picked = rng.choice(rows, size=n, replace=False)
            val[picked, site] = True
    return val


def high_ratio_stratified_mask(
    observed: np.ndarray,
    gt_ratio: np.ndarray,
    frac: float,
    seed: int,
    high_threshold: float,
    high_fraction: float,
) -> np.ndarray:
    """Holdout sample stratified by GT ratio strength.

    Half (``high_fraction``) of the holdout entries are drawn from cells with
    ``gt_ratio >= high_threshold`` and the rest from below-threshold cells, so
    the val mask over-represents the high-signal region of the editing
    distribution. Used to stress-test methods on harder editing patterns.
    """
    rng = np.random.default_rng(seed)
    obs_idx = np.argwhere(observed)
    total = int(obs_idx.shape[0] * frac)
    high_idx = np.argwhere(observed & (gt_ratio >= high_threshold))
    low_idx = np.argwhere(observed & (gt_ratio < high_threshold))

    n_high = min(high_idx.shape[0], int(total * high_fraction))
    n_low = min(low_idx.shape[0], total - n_high)
    val = np.zeros_like(observed, dtype=bool)
    if n_high > 0:
        picked = rng.choice(high_idx.shape[0], size=n_high, replace=False)
        chosen = high_idx[picked]
        val[chosen[:, 0], chosen[:, 1]] = True
    if n_low > 0:
        picked = rng.choice(low_idx.shape[0], size=n_low, replace=False)
        chosen = low_idx[picked]
        val[chosen[:, 0], chosen[:, 1]] = True
    return val


def site_drop_mask(
    observed: np.ndarray,
    gt_ratio: np.ndarray,
    target_frac: float,
    seed: int,
    min_site_observed: int,
    max_sites: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Holdout the entirety of a small set of high-signal sites.

    Ranks sites by ``99th-percentile gt_ratio + 0.001 * coverage`` and selects
    the top sites until either ``target_frac * total_observed_entries`` or
    ``max_sites`` is reached. Returns ``(val_mask, selected_site_indices)``.
    Tests how well a method generalises when entire columns disappear from
    training, rather than scattered cells.
    """
    rng = np.random.default_rng(seed)
    site_support = observed.sum(axis=0)
    site_peak = np.nanpercentile(np.nan_to_num(gt_ratio, nan=0.0), 99, axis=0)
    candidate = np.where(site_support >= min_site_observed)[0]
    if candidate.size == 0:
        return np.zeros_like(observed, dtype=bool), np.array([], dtype=int)

    score = site_peak[candidate] + 0.001 * site_support[candidate]
    candidate = candidate[np.argsort(score)[::-1]]
    target_entries = int(observed.sum() * target_frac)
    selected = []
    total = 0
    for site in candidate:
        selected.append(int(site))
        total += int(site_support[site])
        if total >= target_entries or len(selected) >= max_sites:
            break

    selected_sites = np.array(selected, dtype=int)
    val = np.zeros_like(observed, dtype=bool)
    if selected_sites.size:
        val[:, selected_sites] = observed[:, selected_sites]
    return val, selected_sites


def complete_observed_block(
    observed: np.ndarray,
    gt_ratio: np.ndarray,
    block_spots: int,
    block_sites: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find a sub-block of spots and sites where every cell is observed.

    Picks the highest-coverage spots first (falling back to smaller blocks if
    the requested ``block_spots`` cannot yield enough fully-observed columns)
    and ranks the columns by a combined variance, positive-fraction, and 99th
    percentile score. Returns ``(rows, cols)`` indices defining the block.
    """
    row_cov = observed.sum(axis=1)
    rows = np.argsort(-row_cov)[:block_spots]
    complete_cols = np.where(observed[rows].all(axis=0))[0]
    if complete_cols.size < block_sites:
        for n_rows in [256, 192, 160, 128, 100, 96, 80, 64, 48, 32, 24, 20]:
            rows = np.argsort(-row_cov)[:n_rows]
            complete_cols = np.where(observed[rows].all(axis=0))[0]
            if complete_cols.size >= block_sites:
                break
    if complete_cols.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    gt_block = gt_ratio[np.ix_(rows, complete_cols)]
    score = (
        np.nanvar(gt_block, axis=0)
        + 0.1 * np.nanmean(np.nan_to_num(gt_block, nan=0.0) > 0, axis=0)
        + 0.5 * np.nanpercentile(np.nan_to_num(gt_block, nan=0.0), 99, axis=0)
    )
    cols = complete_cols[np.argsort(score)[-min(block_sites, complete_cols.size):]]
    return np.sort(rows.astype(int)), np.sort(cols.astype(int))


def spatial_block_mask(
    observed: np.ndarray,
    gt_ratio: np.ndarray,
    block_spots: int,
    block_sites: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Holdout an entire dense observed sub-block (rows x cols).

    Wraps :func:`complete_observed_block` and converts the chosen rows/cols
    into a boolean val mask. Returns ``(val_mask, rows, cols)``. Useful for
    evaluating methods on a fully missing-at-random rectangular patch.
    """
    rows, cols = complete_observed_block(observed, gt_ratio, block_spots, block_sites)
    val = np.zeros_like(observed, dtype=bool)
    if rows.size and cols.size:
        val[np.ix_(rows, cols)] = True
        val &= observed
    return val, rows, cols


def train_ratio_from_mask(gt_ratio: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    """Return ``gt_ratio`` masked so non-train cells are NaN (float32)."""
    return np.where(train_mask, gt_ratio, np.nan).astype(np.float32)


def write_index_csv(path: str, values: Iterable[int], column: str) -> None:
    """Write a one-column CSV of integer indices (used for selected_spots / sites)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([column])
        for value in values:
            writer.writerow([int(value)])


def save_mask_case(
    out_dir: str,
    name: str,
    data: Dict[str, np.ndarray],
    val_mask: np.ndarray,
    metadata: Dict,
) -> None:
    """Persist one mask case (train/val/observed/gt) plus a JSON summary.

    Creates ``<out_dir>/<name>/`` containing the four boolean / float arrays,
    optional ``selected_spots.csv`` / ``selected_sites.csv`` index lists, and
    a ``mask_summary.json`` recording mask kind, holdout fractions, depth
    stats, and any case-specific metadata.
    """
    case_dir = os.path.join(out_dir, name)
    os.makedirs(case_dir, exist_ok=True)
    observed = data["observed_mask"]
    train_mask = observed & ~val_mask
    gt_ratio = data["gt_ratio"]
    train_ratio = train_ratio_from_mask(gt_ratio, train_mask)

    np.save(os.path.join(case_dir, "observed_mask.npy"), observed)
    np.save(os.path.join(case_dir, "train_mask.npy"), train_mask)
    np.save(os.path.join(case_dir, "val_mask.npy"), val_mask)
    np.save(os.path.join(case_dir, "gt_ratio.npy"), gt_ratio)
    np.save(os.path.join(case_dir, "train_ratio.npy"), train_ratio)

    selected_spots = metadata.pop("selected_spots", np.array([], dtype=int))
    selected_sites = metadata.pop("selected_sites", np.array([], dtype=int))
    write_index_csv(os.path.join(case_dir, "selected_spots.csv"), selected_spots, "spot_index")
    write_index_csv(os.path.join(case_dir, "selected_sites.csv"), selected_sites, "filtered_site_index")

    summary = {
        "mask_name": name,
        "shape": [int(gt_ratio.shape[0]), int(gt_ratio.shape[1])],
        "observed_entries": int(observed.sum()),
        "train_entries": int(train_mask.sum()),
        "val_entries": int(val_mask.sum()),
        "effective_holdout_frac": float(val_mask.sum() / max(1, observed.sum())),
        "site_filter_threshold_used": int(data["used_threshold"]),
        "depth_stats": data["depth_stats"],
        "selected_spots_count": int(len(selected_spots)),
        "selected_sites_count": int(len(selected_sites)),
        **metadata,
    }
    with open(os.path.join(case_dir, "mask_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    """CLI entry point: write all mask cases for one sample under ``--out_dir``.

    Generates a fixed catalog of mask cases (random_{10,30,50,60},
    per_site_random_{20,30,40,60}, high_ratio_stratified, site_drop,
    spatial_block, mixed_hard) under ``<out_dir>/<case_name>/``. Each case
    saves train/val/observed/gt arrays plus a per-case JSON summary; the
    train/val arrays are guaranteed to satisfy the project's mask contract
    ``train_mask | val_mask == observed_mask``.
    """
    ap = argparse.ArgumentParser(description="Generate shared harder masks for sample 151673.")
    ap.add_argument("--source_root", default=DEFAULT_SOURCE_ROOT)
    ap.add_argument("--h5ad", default=os.path.join(DEFAULT_SOURCE_ROOT, "data", "151673", "adata_ai_compressed.h5ad"))
    ap.add_argument("--out_dir", default=os.path.join(DEFAULT_SOURCE_ROOT, "results", "masks", "masks_151673_seed42"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=int, default=10)
    ap.add_argument("--min_depth", type=int, default=10)
    ap.add_argument("--random_frac", type=float, default=0.10)
    ap.add_argument("--random_mid_frac", type=float, default=0.30,
                    help="mid-strength random mask, useful for ablation studies")
    ap.add_argument("--random_hard_frac", type=float, default=0.50)
    ap.add_argument("--random_xhard_frac", type=float, default=0.60,
                    help="extra-hard random mask using the final 60%% holdout setting")
    ap.add_argument("--high_ratio_threshold", type=float, default=0.05)
    ap.add_argument("--high_ratio_holdout_frac", type=float, default=0.30)
    ap.add_argument("--site_drop_frac", type=float, default=0.05)
    ap.add_argument("--site_drop_min_observed", type=int, default=20)
    ap.add_argument("--site_drop_max_sites", type=int, default=50)
    ap.add_argument("--block_spots", type=int, default=100)
    ap.add_argument("--block_sites", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = load_filtered_data(args.source_root, args.h5ad, args.threshold, args.min_depth)
    observed = data["observed_mask"]
    gt_ratio = data["gt_ratio"]

    save_mask_case(
        args.out_dir,
        f"random_10_seed{args.seed}",
        data,
        random_entry_mask(observed, args.random_frac, args.seed),
        {"kind": "random_entry", "seed": args.seed, "holdout_frac": args.random_frac},
    )
    save_mask_case(
        args.out_dir,
        f"random_30_seed{args.seed}",
        data,
        random_entry_mask(observed, args.random_mid_frac, args.seed),
        {"kind": "random_entry", "seed": args.seed, "holdout_frac": args.random_mid_frac},
    )
    save_mask_case(
        args.out_dir,
        f"random_50_seed{args.seed}",
        data,
        random_entry_mask(observed, args.random_hard_frac, args.seed),
        {"kind": "random_entry", "seed": args.seed, "holdout_frac": args.random_hard_frac},
    )
    save_mask_case(
        args.out_dir,
        f"random_60_seed{args.seed}",
        data,
        random_entry_mask(observed, args.random_xhard_frac, args.seed),
        {"kind": "random_entry", "seed": args.seed, "holdout_frac": args.random_xhard_frac},
    )
    save_mask_case(
        args.out_dir,
        f"per_site_random_20_seed{args.seed}",
        data,
        per_site_mask(observed, 0.20, args.seed, min_train_per_site=1),
        {"kind": "per_site_random", "seed": args.seed, "holdout_frac_per_site": 0.20},
    )
    save_mask_case(
        args.out_dir,
        f"per_site_random_30_seed{args.seed}",
        data,
        per_site_mask(observed, args.high_ratio_holdout_frac, args.seed, min_train_per_site=1),
        {"kind": "per_site_random", "seed": args.seed, "holdout_frac_per_site": args.high_ratio_holdout_frac},
    )
    save_mask_case(
        args.out_dir,
        f"per_site_random_40_seed{args.seed}",
        data,
        per_site_mask(observed, 0.40, args.seed, min_train_per_site=1),
        {"kind": "per_site_random", "seed": args.seed, "holdout_frac_per_site": 0.40},
    )
    save_mask_case(
        args.out_dir,
        f"per_site_random_60_seed{args.seed}",
        data,
        per_site_mask(observed, 0.60, args.seed, min_train_per_site=1),
        {"kind": "per_site_random", "seed": args.seed, "holdout_frac_per_site": 0.60},
    )
    save_mask_case(
        args.out_dir,
        f"high_ratio_stratified_seed{args.seed}",
        data,
        high_ratio_stratified_mask(
            observed,
            gt_ratio,
            frac=args.high_ratio_holdout_frac,
            seed=args.seed,
            high_threshold=args.high_ratio_threshold,
            high_fraction=0.5,
        ),
        {
            "kind": "high_ratio_stratified",
            "seed": args.seed,
            "overall_holdout_frac": args.high_ratio_holdout_frac,
            "high_ratio_threshold": args.high_ratio_threshold,
            "target_high_fraction_inside_holdout": 0.5,
        },
    )

    site_val, selected_sites = site_drop_mask(
        observed,
        gt_ratio,
        target_frac=args.site_drop_frac,
        seed=args.seed,
        min_site_observed=args.site_drop_min_observed,
        max_sites=args.site_drop_max_sites,
    )
    save_mask_case(
        args.out_dir,
        f"site_drop_seed{args.seed}",
        data,
        site_val,
        {
            "kind": "site_drop",
            "seed": args.seed,
            "target_frac": args.site_drop_frac,
            "min_site_observed": args.site_drop_min_observed,
            "selected_sites": selected_sites,
        },
    )

    block_val, rows, cols = spatial_block_mask(observed, gt_ratio, args.block_spots, args.block_sites)
    save_mask_case(
        args.out_dir,
        f"spatial_block_seed{args.seed}",
        data,
        block_val,
        {
            "kind": "spatial_block",
            "seed": args.seed,
            "requested_block_spots": args.block_spots,
            "requested_block_sites": args.block_sites,
            "selected_spots": rows,
            "selected_sites": cols,
        },
    )

    mixed = random_entry_mask(observed, args.random_hard_frac, args.seed)
    mixed |= high_ratio_stratified_mask(observed, gt_ratio, 0.10, args.seed + 17, args.high_ratio_threshold, 0.7)
    mixed |= site_val
    mixed |= block_val
    mixed &= observed
    save_mask_case(
        args.out_dir,
        f"mixed_hard_seed{args.seed}",
        data,
        mixed,
        {
            "kind": "mixed_hard",
            "seed": args.seed,
            "components": ["random_50", "high_ratio_extra", "site_drop", "spatial_block"],
            "selected_spots": rows,
            "selected_sites": np.unique(np.concatenate([selected_sites, cols])).astype(int),
        },
    )

    print(json.dumps({"out_dir": os.path.abspath(args.out_dir)}, indent=2))


if __name__ == "__main__":
    main()
