import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "models", "AIRGate-ST"))

from shared.data_utils import _cluster_order  # noqa: E402


def draw_panels_gray_nan(arrays, titles, col_order, save_path, suptitle="",
                         vmin=0.0, vmax=1.0):
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("lightgray")

    n = len(arrays)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 7), constrained_layout=True)
    if n == 1:
        axes = [axes]

    im = None
    for ax, arr, title in zip(axes, arrays, titles):
        ordered = arr[:, col_order]
        im = ax.imshow(ordered, aspect="auto", cmap=cmap,
                       vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    if im is not None:
        cbar = fig.colorbar(im, ax=axes, shrink=0.8)
        cbar.set_label("A-to-I editing ratio")

    if suptitle:
        fig.suptitle(suptitle, fontsize=9, y=1.01)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] saved -> {save_path}")


def _mean_column_order(gt_sub: np.ndarray) -> np.ndarray:
    col_mean = np.nanmean(gt_sub, axis=0)
    col_mean = np.nan_to_num(col_mean, nan=-np.inf)
    return np.argsort(-col_mean)


def _score_sites(gt_block: np.ndarray,
                 holdout_block: np.ndarray,
                 mode: str,
                 high_signal_threshold: float = 0.05) -> np.ndarray:
    var_score = np.nanvar(gt_block, axis=0)
    nz_score = np.nanmean(np.nan_to_num(gt_block, nan=0.0) > 0, axis=0)
    holdout_score = holdout_block.sum(axis=0).astype(np.float32)

    if mode == "dense":
        return var_score + 0.1 * nz_score + 0.01 * holdout_score

    q95_score = np.nanquantile(gt_block, 0.95, axis=0)
    max_score = np.nanmax(gt_block, axis=0)
    if mode == "high_signal":
        high = gt_block >= high_signal_threshold
        high_frac = np.nanmean(high, axis=0)
        high_count = high.sum(axis=0)
        high_sum = np.nansum(np.where(high, gt_block, 0.0), axis=0)
        high_mean = np.divide(
            high_sum,
            high_count,
            out=np.zeros_like(high_sum, dtype=np.float64),
            where=high_count > 0,
        )
        return 8.0 * high_frac + 2.0 * high_mean + 0.5 * q95_score + 0.1 * max_score + 0.01 * holdout_score

    # Rare-peak block: prioritize strong peaks, but avoid selecting columns
    # that are effectively all-zero by keeping a small nonzero component.
    return 1.5 * q95_score + 1.0 * max_score + 0.4 * var_score + 0.1 * nz_score + 0.01 * holdout_score


def _rank_rows(gt_ratio: np.ndarray,
               mode: str,
               high_signal_threshold: float = 0.05) -> np.ndarray:
    observed_mask = ~np.isnan(gt_ratio)
    row_cov = observed_mask.sum(axis=1).astype(np.float32)
    row_vals = np.nan_to_num(gt_ratio, nan=0.0)

    if mode == "dense":
        score = row_cov
    elif mode == "high_signal":
        high_count = (row_vals >= high_signal_threshold).sum(axis=1).astype(np.float32)
        high_sum = np.where(row_vals >= high_signal_threshold, row_vals, 0.0).sum(axis=1)
        score = 0.1 * row_cov + 200.0 * high_count + 1000.0 * high_sum
    else:
        peak99 = np.quantile(row_vals, 0.99, axis=1)
        peak95 = np.quantile(row_vals, 0.95, axis=1)
        nz_count = (row_vals > 0).sum(axis=1).astype(np.float32)
        # Coverage still matters, but rows containing stronger peaks are promoted.
        score = row_cov + 250.0 * peak99 + 100.0 * peak95 + 0.02 * nz_count
    return np.argsort(-score)


def _candidate_row_counts(min_spots: int, max_spots: int) -> List[int]:
    base = [1024, 768, 512, 384, 256, 192, 160, 128, 96, 80, 64, 48, 32, 24, 16, 12, 10, 8]
    kept = [n for n in base if min_spots <= n <= max_spots]
    if min_spots not in kept and min_spots <= max_spots:
        kept.append(min_spots)
    return sorted(set(kept), reverse=True)


def select_block(gt_ratio: np.ndarray,
                 val_mask: np.ndarray,
                 top_k_sites: int,
                 min_spots: int,
                 max_spots: int,
                 mode: str,
                 high_signal_threshold: float = 0.05) -> Dict[str, np.ndarray]:
    observed_mask = ~np.isnan(gt_ratio)
    row_order = _rank_rows(gt_ratio, mode=mode, high_signal_threshold=high_signal_threshold)

    best = None
    for n_spots in _candidate_row_counts(min_spots=min_spots, max_spots=max_spots):
        rows = np.sort(row_order[:n_spots])
        fully_obs_cols = np.where(observed_mask[rows].all(axis=0))[0]
        if fully_obs_cols.size < top_k_sites:
            continue

        gt_block = gt_ratio[np.ix_(rows, fully_obs_cols)]
        holdout_block = val_mask[np.ix_(rows, fully_obs_cols)]
        score = _score_sites(
            gt_block,
            holdout_block,
            mode=mode,
            high_signal_threshold=high_signal_threshold,
        )
        cols = fully_obs_cols[np.argsort(score)[-top_k_sites:]]

        gt_for_cols = gt_ratio[np.ix_(rows, cols)]
        col_local = _cluster_order(gt_for_cols, axis=1)
        cols = cols[col_local]
        gt_for_rows = gt_ratio[np.ix_(rows, cols)]
        row_local = _cluster_order(gt_for_rows, axis=0)
        rows = rows[row_local]

        best = {
            "rows": rows,
            "cols": cols,
            "fully_observed_cols": np.array([fully_obs_cols.size], dtype=np.int32),
            "selected_n_spots": np.array([n_spots], dtype=np.int32),
        }
        break

    if best is None:
        raise RuntimeError(
            f"Could not build a {mode} block with top_k_sites={top_k_sites} "
            f"and min_spots={min_spots}."
        )
    return best


def render_block(name: str,
                 out_dir: str,
                 gt_ratio: np.ndarray,
                 train_ratio: np.ndarray,
                 pred_ratio: np.ndarray,
                 val_mask: np.ndarray,
                 rows: np.ndarray,
                 cols: np.ndarray) -> Dict[str, object]:
    os.makedirs(out_dir, exist_ok=True)

    gt_sub = gt_ratio[np.ix_(rows, cols)]
    train_sub = train_ratio[np.ix_(rows, cols)]
    pred_sub = pred_ratio[np.ix_(rows, cols)]
    gt_hold = np.where(val_mask, gt_ratio, np.nan)[np.ix_(rows, cols)]
    pred_hold = np.where(val_mask, pred_ratio, np.nan)[np.ix_(rows, cols)]
    col_idx = _mean_column_order(gt_sub)

    full_path = os.path.join(out_dir, f"{name}_full_heatmap.png")
    holdout_path = os.path.join(out_dir, f"{name}_holdout_heatmap.png")

    draw_panels_gray_nan(
        arrays=[gt_sub, train_sub, pred_sub],
        titles=[
            f"{name}: GT",
            f"{name}: Masked Train (gray = held out)",
            f"{name}: Pred",
        ],
        col_order=col_idx,
        save_path=full_path,
        suptitle=f"{name} observed-only block: {len(rows)} spots x {len(cols)} sites",
    )

    draw_panels_gray_nan(
        arrays=[gt_hold, pred_hold],
        titles=[
            f"{name}: GT @ holdout",
            f"{name}: Pred @ holdout",
        ],
        col_order=col_idx,
        save_path=holdout_path,
        suptitle=f"{name} holdout-only block | same sub-matrix | gray = non-holdout",
    )


    return {
        "rows": int(len(rows)),
        "cols": int(len(cols)),
        "column_order": "gt_mean_desc",
        "full_heatmap": full_path,
        "holdout_heatmap": holdout_path,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Render dense, rare-peak, and high-signal observed-only heatmaps.")
    ap.add_argument("--model_dir", default=os.path.join("results", "phase1_harder_benchmark", "AIRGate-ST", "seed42"))
    ap.add_argument("--out_dir", default=os.path.join("results", "visualization", "airgate_dual_block"))
    ap.add_argument("--top_k_sites", type=int, default=50)
    ap.add_argument("--dense_min_spots", type=int, default=128)
    ap.add_argument("--dense_max_spots", type=int, default=1024)
    ap.add_argument("--peak_min_spots", type=int, default=32)
    ap.add_argument("--peak_max_spots", type=int, default=256)
    ap.add_argument("--high_signal_min_spots", type=int, default=32)
    ap.add_argument("--high_signal_max_spots", type=int, default=64)
    ap.add_argument("--high_signal_threshold", type=float, default=0.05)
    args = ap.parse_args()

    model_dir = os.path.join(REPO_ROOT, args.model_dir)
    out_dir = os.path.join(REPO_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    gt_ratio = np.load(os.path.join(model_dir, "gt_ratio.npy")).astype(np.float32)
    train_ratio = np.load(os.path.join(model_dir, "train_ratio.npy")).astype(np.float32)
    pred_ratio = np.load(os.path.join(model_dir, "pred_ratio.npy")).astype(np.float32)
    val_mask = np.load(os.path.join(model_dir, "val_mask.npy")).astype(bool)

    dense = select_block(
        gt_ratio=gt_ratio,
        val_mask=val_mask,
        top_k_sites=args.top_k_sites,
        min_spots=args.dense_min_spots,
        max_spots=args.dense_max_spots,
        mode="dense",
        high_signal_threshold=args.high_signal_threshold,
    )
    peak = select_block(
        gt_ratio=gt_ratio,
        val_mask=val_mask,
        top_k_sites=args.top_k_sites,
        min_spots=args.peak_min_spots,
        max_spots=args.peak_max_spots,
        mode="peak",
        high_signal_threshold=args.high_signal_threshold,
    )
    high_signal = select_block(
        gt_ratio=gt_ratio,
        val_mask=val_mask,
        top_k_sites=args.top_k_sites,
        min_spots=args.high_signal_min_spots,
        max_spots=args.high_signal_max_spots,
        mode="high_signal",
        high_signal_threshold=args.high_signal_threshold,
    )

    dense_info = render_block(
        name="dense_block",
        out_dir=out_dir,
        gt_ratio=gt_ratio,
        train_ratio=train_ratio,
        pred_ratio=pred_ratio,
        val_mask=val_mask,
        rows=dense["rows"],
        cols=dense["cols"],
    )
    peak_info = render_block(
        name="rare_peak_block",
        out_dir=out_dir,
        gt_ratio=gt_ratio,
        train_ratio=train_ratio,
        pred_ratio=pred_ratio,
        val_mask=val_mask,
        rows=peak["rows"],
        cols=peak["cols"],
    )
    high_signal_info = render_block(
        name=f"high_signal_ge_{args.high_signal_threshold:g}_block",
        out_dir=out_dir,
        gt_ratio=gt_ratio,
        train_ratio=train_ratio,
        pred_ratio=pred_ratio,
        val_mask=val_mask,
        rows=high_signal["rows"],
        cols=high_signal["cols"],
    )

    summary = {
        "model_dir": model_dir,
        "out_dir": out_dir,
        "top_k_sites": int(args.top_k_sites),
        "dense_block": {
            "selected_n_spots": int(dense["selected_n_spots"][0]),
            "available_fully_observed_cols": int(dense["fully_observed_cols"][0]),
            **dense_info,
        },
        "rare_peak_block": {
            "selected_n_spots": int(peak["selected_n_spots"][0]),
            "available_fully_observed_cols": int(peak["fully_observed_cols"][0]),
            **peak_info,
        },
        "high_signal_block": {
            "threshold": float(args.high_signal_threshold),
            "selected_n_spots": int(high_signal["selected_n_spots"][0]),
            "available_fully_observed_cols": int(high_signal["fully_observed_cols"][0]),
            **high_signal_info,
        },
    }

    summary_path = os.path.join(out_dir, "dual_block_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
