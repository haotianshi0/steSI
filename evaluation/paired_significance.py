"""Paired-bootstrap significance tests between models on shared holdout cells.

For every (layer, metric) pair, compares each non-baseline method against a
designated baseline (default: Mean Baseline) using their predictions on the
same set of valid holdout cells. The test resamples cells with replacement,
recomputes the metric difference per replicate, and reports an empirical
two-sided 95% confidence interval and a p-value approximating
``P(model worse than baseline)`` under the bootstrap distribution.

Cells where either prediction or the ground truth is non-finite are dropped
before bootstrapping. Output is one row per ``(method, layer, metric)``
combination, written as both CSV and JSON for downstream tabling.
"""

import argparse
import csv
import json
import os
import sys
from typing import Dict, List

import numpy as np


_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from collect_layered_metrics import layer_masks, read_model_registry  # noqa: E402


def metric_scores(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Compute RMSE, MAE, cosine, Pearson r on flat 1-D paired arrays."""
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)
    rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
    mae = float(np.mean(np.abs(pred - gt)))
    denom = float(np.linalg.norm(pred) * np.linalg.norm(gt))
    cos = float(np.dot(pred, gt) / (denom + 1e-12)) if denom > 1e-12 else 0.0
    if np.std(pred) > 1e-12 and np.std(gt) > 1e-12:
        pearson = float(np.corrcoef(pred, gt)[0, 1])
    else:
        pearson = 0.0
    return {"rmse": rmse, "mae": mae, "cos": cos, "pearson_r": pearson}


def paired_diffs(model_pred: np.ndarray, base_pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Return (model_metric - baseline_metric) for each of the four metrics."""
    model_scores = metric_scores(model_pred, gt)
    base_scores = metric_scores(base_pred, gt)
    return {
        "rmse": model_scores["rmse"] - base_scores["rmse"],
        "mae": model_scores["mae"] - base_scores["mae"],
        "cos": model_scores["cos"] - base_scores["cos"],
        "pearson_r": model_scores["pearson_r"] - base_scores["pearson_r"],
    }


def bootstrap_metric(
    model_pred: np.ndarray,
    base_pred: np.ndarray,
    gt: np.ndarray,
    metric: str,
    iters: int,
    sample_size: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    """Run a paired bootstrap and return ``(observed_diff, ci95_low, ci95_high, p_better)``.

    ``p_better`` uses the appropriate inequality for the metric direction
    (RMSE/MAE: lower is better; cosine/PCC: higher is better) and includes
    the standard +1 smoothing to avoid zero-probability outputs at small
    iteration counts.
    """
    observed = paired_diffs(model_pred, base_pred, gt)[metric]
    n = gt.shape[0]
    boot_n = min(n, sample_size) if sample_size > 0 else n
    boot = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        idx = rng.integers(0, n, size=boot_n)
        boot[i] = paired_diffs(model_pred[idx], base_pred[idx], gt[idx])[metric]

    lo, hi = np.percentile(boot, [2.5, 97.5])
    if metric in ("rmse", "mae"):
        # Negative difference means model has lower error than baseline.
        p_better = float((np.count_nonzero(boot >= 0.0) + 1) / (iters + 1))
    else:
        # Positive difference means model has higher similarity/correlation.
        p_better = float((np.count_nonzero(boot <= 0.0) + 1) / (iters + 1))
    return float(observed), float(lo), float(hi), p_better


def main() -> None:
    """CLI entry point: run the paired bootstrap for every method against ``--baseline``.

    Loads the GT, val mask, and per-method predictions from the run-specific
    ``model_registry.csv``, builds the same layered evaluation as
    ``collect_layered_metrics.py``, and writes
    ``paired_significance.csv`` / ``paired_significance.json`` under
    ``--out_dir``.
    """
    parser = argparse.ArgumentParser(description="Paired bootstrap significance tests for models sharing the same holdout cells.")
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--model_registry", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--baseline", default="Mean Baseline")
    parser.add_argument("--bootstrap_iters", type=int, default=1000)
    parser.add_argument("--bootstrap_sample_size", type=int, default=200000,
                        help="Max paired cells sampled per bootstrap replicate; <=0 uses all valid cells.")
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    gt = np.load(os.path.join(args.mask_dir, "gt_ratio.npy")).astype(np.float32)
    val_mask = np.load(os.path.join(args.mask_dir, "val_mask.npy")).astype(bool)
    layers = layer_masks(gt, val_mask, args.mask_dir)
    models = read_model_registry(args.model_registry)
    pred_by_method = {}
    for model in models:
        pred_by_method[model["method"]] = np.load(model["pred_path"]).astype(np.float32)

    if args.baseline not in pred_by_method:
        raise ValueError(f"Baseline not found in registry: {args.baseline}")

    rng = np.random.default_rng(args.seed)
    rows: List[Dict[str, object]] = []
    baseline_pred = pred_by_method[args.baseline]
    for layer_name, layer_mask in layers:
        for method, pred in pred_by_method.items():
            if method == args.baseline:
                continue
            valid = layer_mask & np.isfinite(gt) & np.isfinite(baseline_pred) & np.isfinite(pred)
            n = int(valid.sum())
            if n < 3:
                continue
            gt_v = gt[valid]
            base_v = baseline_pred[valid]
            pred_v = pred[valid]
            for metric in ["rmse", "mae", "cos", "pearson_r"]:
                diff, ci_low, ci_high, p_better = bootstrap_metric(
                    pred_v,
                    base_v,
                    gt_v,
                    metric,
                    args.bootstrap_iters,
                    args.bootstrap_sample_size,
                    rng,
                )
                rows.append({
                    "baseline": args.baseline,
                    "method": method,
                    "layer": layer_name,
                    "metric": metric,
                    "n": n,
                    "diff_model_minus_baseline": diff,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "p_model_better": p_better,
                    "better_direction": "lower" if metric in ("rmse", "mae") else "higher",
                    "bootstrap_iters": args.bootstrap_iters,
                    "bootstrap_sample_size": min(n, args.bootstrap_sample_size) if args.bootstrap_sample_size > 0 else n,
                })

    out_csv = os.path.join(args.out_dir, "paired_significance.csv")
    fieldnames = [
        "baseline",
        "method",
        "layer",
        "metric",
        "n",
        "diff_model_minus_baseline",
        "ci95_low",
        "ci95_high",
        "p_model_better",
        "better_direction",
        "bootstrap_iters",
        "bootstrap_sample_size",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    out_json = os.path.join(args.out_dir, "paired_significance.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, indent=2)

    print(json.dumps({
        "significance_csv": os.path.abspath(out_csv),
        "significance_json": os.path.abspath(out_json),
        "baseline": args.baseline,
        "n_tests": len(rows),
    }, indent=2))


if __name__ == "__main__":
    main()
