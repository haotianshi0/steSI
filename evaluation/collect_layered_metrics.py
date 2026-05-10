"""Compute layered imputation metrics from saved prediction arrays.

The evaluator only scores held-out observed entries and validates the core
NaN/zero contract: unobserved values are not silently treated as zeros, and
holdout entries used for metrics must have finite predictions.
"""

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np


DEFAULT_SOURCE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def add_source_paths(source_root: str) -> None:
    """Add the project's helper directory to ``sys.path`` (kept for symmetry)."""
    sys.path.insert(0, os.path.join(source_root, "src", "models", "AIRGate-ST"))


def metric_values(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Compute RMSE, MAE, cosine similarity, Pearson r over ``mask`` cells.

    Returns a dict with ``n`` (cell count) and the four scalar metrics. Cells
    where either ``pred`` or ``gt`` is non-finite are dropped before scoring.
    Pearson r and cosine fall back to 0 if either vector has zero variance.
    """
    valid = mask & np.isfinite(gt) & np.isfinite(pred)
    if valid.sum() == 0:
        return {"n": 0, "rmse": float("nan"), "mae": float("nan"), "cos": float("nan"), "pearson_r": float("nan")}

    y = gt[valid].astype(np.float64)
    p = pred[valid].astype(np.float64)
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    mae = float(np.mean(np.abs(p - y)))
    if np.std(y) > 1e-12 and np.std(p) > 1e-12:
        r = float(np.corrcoef(p, y)[0, 1])
    else:
        r = 0.0
    denom = float(np.linalg.norm(y) * np.linalg.norm(p))
    cos = float(np.dot(y, p) / (denom + 1e-12)) if denom > 1e-12 else 0.0
    return {"n": int(valid.sum()), "rmse": rmse, "mae": mae, "cos": cos, "pearson_r": r}


def prediction_coverage(
    pred: np.ndarray,
    observed_mask: np.ndarray,
    train_mask: np.ndarray | None,
    val_mask: np.ndarray,
) -> Dict[str, float]:
    """Audit a prediction matrix for coverage and NaN/zero hygiene.

    Counts cells that should be finite but are NaN (observed/train/holdout),
    cells where the model extrapolated into the unobserved region (which is
    allowed but not used for scoring), and several "did the model output
    something biologically meaningful at unobserved sites" diagnostics
    (fraction > 0.05/0.10, fraction collapsed to ~0). Used for sanity, not
    grading.
    """
    pred_finite = np.isfinite(pred)
    unobserved = ~observed_mask
    all_unobserved_cols = observed_mask.sum(axis=0) == 0
    partial_unobserved = unobserved.copy()
    if np.any(all_unobserved_cols):
        partial_unobserved[:, all_unobserved_cols] = False
    out = {
        "observed_to_nan": int((observed_mask & ~pred_finite).sum()),
        "train_to_nan": int((train_mask & ~pred_finite).sum()) if train_mask is not None else 0,
        "holdout_to_nan": int((val_mask & ~pred_finite).sum()),
        "unobserved_to_pred": int((unobserved & pred_finite).sum()),
        "unobserved_to_nan": int((unobserved & ~pred_finite).sum()),
        "partial_unobserved_to_pred": int((partial_unobserved & pred_finite).sum()),
        "partial_unobserved_to_nan": int((partial_unobserved & ~pred_finite).sum()),
        "all_unobserved_site_count": int(all_unobserved_cols.sum()),
        "all_unobserved_site_pred_all_nan_count": int(
            np.all(~pred_finite[:, all_unobserved_cols], axis=0).sum()
        ) if np.any(all_unobserved_cols) else 0,
        "all_unobserved_site_pred_any_finite_count": int(
            np.any(pred_finite[:, all_unobserved_cols], axis=0).sum()
        ) if np.any(all_unobserved_cols) else 0,
        "unobserved_high_frac_gt_0p05": float((pred[unobserved & pred_finite] > 0.05).mean()) if np.any(unobserved & pred_finite) else float("nan"),
        "unobserved_high_frac_gt_0p10": float((pred[unobserved & pred_finite] > 0.10).mean()) if np.any(unobserved & pred_finite) else float("nan"),
        "unobserved_mass_at_zero_lt_1e_3": float((pred[unobserved & pred_finite] < 1e-3).mean()) if np.any(unobserved & pred_finite) else float("nan"),
    }
    return out


def require_semantics(name: str, condition: bool, detail: str) -> None:
    """Raise ``ValueError(f'{name}: {detail}')`` unless ``condition`` holds."""
    if not condition:
        raise ValueError(f"{name}: {detail}")


def validate_nan_zero_inputs(gt: np.ndarray, observed_mask: np.ndarray, val_mask: np.ndarray, mask_dir: str) -> None:
    """Enforce the project's NaN/zero contract on the saved mask + GT arrays.

    Confirms (a) every observed cell has a finite ground-truth ratio, (b) no
    unobserved cell has a finite ground-truth value, and (c) when
    ``train_mask.npy`` / ``train_ratio.npy`` exist, that ``train_mask | val_mask
    == observed_mask``, train and val do not overlap, and the train ratio is
    finite exactly on train cells.
    """
    gt_finite = np.isfinite(gt)
    require_semantics(
        "gt_ratio",
        not np.any(observed_mask & ~gt_finite),
        f"{int((observed_mask & ~gt_finite).sum())} observed entries are NaN",
    )
    require_semantics(
        "gt_ratio",
        not np.any((~observed_mask) & gt_finite),
        f"{int(((~observed_mask) & gt_finite).sum())} unobserved entries are finite",
    )

    train_mask_path = os.path.join(mask_dir, "train_mask.npy")
    train_ratio_path = os.path.join(mask_dir, "train_ratio.npy")
    if os.path.exists(train_mask_path):
        train_mask = np.load(train_mask_path).astype(bool)
        require_semantics("train_mask", train_mask.shape == observed_mask.shape, "shape mismatch")
        require_semantics(
            "train/val masks",
            not np.any(train_mask & val_mask),
            f"{int((train_mask & val_mask).sum())} entries are both train and val",
        )
        require_semantics(
            "observed_mask",
            np.array_equal(train_mask | val_mask, observed_mask),
            "must equal train_mask | val_mask",
        )

        if os.path.exists(train_ratio_path):
            train_ratio = np.load(train_ratio_path).astype(np.float32)
            train_finite = np.isfinite(train_ratio)
            require_semantics("train_ratio", train_ratio.shape == observed_mask.shape, "shape mismatch")
            require_semantics(
                "train_ratio",
                not np.any(train_mask & ~train_finite),
                f"{int((train_mask & ~train_finite).sum())} train entries are NaN",
            )
            require_semantics(
                "train_ratio",
                not np.any((~train_mask) & train_finite),
                f"{int(((~train_mask) & train_finite).sum())} non-train entries are finite",
            )


def validate_nan_zero_prediction(
    method: str,
    pred: np.ndarray,
    gt: np.ndarray,
    train_mask: np.ndarray | None,
    val_mask: np.ndarray,
) -> None:
    """Enforce the same NaN/zero contract on a model's prediction matrix.

    Checks finite predictions on all observed (train + val) cells, that all
    finite predictions stay within ``[0, 1]``, that holdout cells are finite,
    and (when ``train_mask`` is provided) that train cells exactly preserve
    the ground-truth ratio.
    """
    pred_finite = np.isfinite(pred)
    require_semantics(
        method,
        not np.any((train_mask | val_mask) & ~pred_finite) if train_mask is not None else not np.any(val_mask & ~pred_finite),
        "observed metric/training entries contain NaN/Inf",
    )
    finite_pred = pred[pred_finite]
    require_semantics(
        method,
        finite_pred.size == 0 or not np.any((finite_pred < 0.0) | (finite_pred > 1.0)),
        "finite prediction values outside [0, 1]",
    )
    require_semantics(
        method,
        not np.any(val_mask & ~pred_finite),
        f"{int((val_mask & ~pred_finite).sum())} metric/holdout entries are NaN",
    )
    if train_mask is not None:
        require_semantics(
            method,
            np.allclose(pred[train_mask], gt[train_mask], equal_nan=False),
            "training-visible gt_ratio values were overwritten",
        )


def read_model_registry(path: str) -> List[Dict[str, str]]:
    """Read a ``method,pred_path`` CSV into a list of dicts."""
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({"method": row["method"], "pred_path": row["pred_path"]})
    return rows


def read_index_csv(path: str) -> np.ndarray:
    """Read a one-column index CSV (selected_sites/selected_spots) into ``int`` array."""
    values = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            value = next(iter(row.values()))
            if value not in (None, ""):
                values.append(int(value))
    return np.asarray(values, dtype=int)


def default_registry(source_root: str) -> List[Dict[str, str]]:
    """Refuse to fall back to a stale built-in registry; force ``--model_registry``."""
    raise ValueError(
        "No default model registry is used because stale result paths are unsafe. "
        "Pass --model_registry pointing to the run-specific model_registry.csv."
    )


def layer_masks(gt: np.ndarray, val_mask: np.ndarray, mask_dir: str) -> List[Tuple[str, np.ndarray]]:
    """Build the list of ``(layer_name, layer_mask)`` pairs used for scoring.

    Always includes ``all_holdout``, ``gt_positive``, ``gt_ge_0p05``, and
    ``gt_ge_0p10``. If the mask directory contains ``selected_sites.csv`` or
    ``selected_spots.csv`` (from the harder mask catalogue), additional
    layers restricted to those rows/columns are appended for finer-grained
    reporting.
    """
    layers = [
        ("all_holdout", val_mask),
        ("gt_positive", val_mask & (gt > 0)),
        ("gt_ge_0p05", val_mask & (gt >= 0.05)),
        ("gt_ge_0p10", val_mask & (gt >= 0.10)),
    ]

    selected_sites_path = os.path.join(mask_dir, "selected_sites.csv")
    selected_spots_path = os.path.join(mask_dir, "selected_spots.csv")
    if os.path.exists(selected_sites_path):
        site_ids = read_index_csv(selected_sites_path)
        if site_ids.size:
            site_layer = np.zeros_like(val_mask, dtype=bool)
            site_layer[:, site_ids] = val_mask[:, site_ids]
            layers.append(("selected_sites_only", site_layer))
    if os.path.exists(selected_spots_path):
        spot_ids = read_index_csv(selected_spots_path)
        if spot_ids.size:
            spot_layer = np.zeros_like(val_mask, dtype=bool)
            spot_layer[spot_ids, :] = val_mask[spot_ids, :]
            layers.append(("selected_spots_only", spot_layer))
            if os.path.exists(selected_sites_path):
                block_layer = np.zeros_like(val_mask, dtype=bool)
                block_layer[np.ix_(spot_ids, site_ids)] = val_mask[np.ix_(spot_ids, site_ids)]
                layers.append(("selected_block_only", block_layer))
    return layers


def main() -> None:
    """CLI entry point: validate inputs, score every method on every layer, write outputs.

    Workflow:

      1. Load ``gt_ratio.npy``, ``val_mask.npy``, ``observed_mask.npy`` from
         ``--mask_dir`` and run :func:`validate_nan_zero_inputs`.
      2. Read the run-specific ``model_registry.csv`` from
         ``--model_registry``. Each row maps a method name to its
         ``pred_ratio.npy`` path.
      3. For each method, load and validate its prediction matrix, then
         score it on every layer returned by :func:`layer_masks`.
      4. Write ``layered_metrics.csv``, ``prediction_coverage.csv``, and a
         combined ``layered_metrics.json`` under ``--out_dir``.
    """
    ap = argparse.ArgumentParser(description="Collect layered holdout and unobserved safety metrics.")
    ap.add_argument("--source_root", default=DEFAULT_SOURCE_ROOT)
    ap.add_argument("--mask_dir", required=True, help="Directory containing gt_ratio.npy, val_mask.npy, observed_mask.npy.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_registry", default=None, help="CSV with columns method,pred_path. Required for non-stale evaluation.")
    args = ap.parse_args()

    add_source_paths(args.source_root)
    os.makedirs(args.out_dir, exist_ok=True)

    gt = np.load(os.path.join(args.mask_dir, "gt_ratio.npy")).astype(np.float32)
    val_mask = np.load(os.path.join(args.mask_dir, "val_mask.npy")).astype(bool)
    observed_mask = np.load(os.path.join(args.mask_dir, "observed_mask.npy")).astype(bool)
    validate_nan_zero_inputs(gt, observed_mask, val_mask, args.mask_dir)
    train_mask_path = os.path.join(args.mask_dir, "train_mask.npy")
    train_mask = np.load(train_mask_path).astype(bool) if os.path.exists(train_mask_path) else None
    models = read_model_registry(args.model_registry) if args.model_registry else default_registry(args.source_root)

    rows = []
    coverage_rows = []
    layers = layer_masks(gt, val_mask, args.mask_dir)
    for model in models:
        pred_path = model["pred_path"]
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"Missing prediction for {model['method']}: {pred_path}")
        pred = np.load(pred_path).astype(np.float32)
        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch for {model['method']}: pred={pred.shape}, gt={gt.shape}")
        validate_nan_zero_prediction(model["method"], pred, gt, train_mask, val_mask)

        for layer_name, layer_mask in layers:
            vals = metric_values(pred, gt, layer_mask)
            rows.append({
                "method": model["method"],
                "layer": layer_name,
                **vals,
                "pred_path": os.path.abspath(pred_path),
                "mask_dir": os.path.abspath(args.mask_dir),
            })
        coverage_rows.append({
            "method": model["method"],
            **prediction_coverage(pred, observed_mask, train_mask, val_mask),
            "pred_path": os.path.abspath(pred_path),
        })

    metrics_csv = os.path.join(args.out_dir, "layered_metrics.csv")
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["method", "layer", "n", "rmse", "mae", "cos", "pearson_r", "pred_path", "mask_dir"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    coverage_csv = os.path.join(args.out_dir, "prediction_coverage.csv")
    with open(coverage_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "method",
            "observed_to_nan",
            "train_to_nan",
            "holdout_to_nan",
            "unobserved_to_pred",
            "unobserved_to_nan",
            "partial_unobserved_to_pred",
            "partial_unobserved_to_nan",
            "all_unobserved_site_count",
            "all_unobserved_site_pred_all_nan_count",
            "all_unobserved_site_pred_any_finite_count",
            "unobserved_high_frac_gt_0p05",
            "unobserved_high_frac_gt_0p10",
            "unobserved_mass_at_zero_lt_1e_3",
            "pred_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(coverage_rows)

    with open(os.path.join(args.out_dir, "layered_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"metrics": rows, "prediction_coverage": coverage_rows}, f, indent=2)

    print(json.dumps({
        "metrics_csv": os.path.abspath(metrics_csv),
        "coverage_csv": os.path.abspath(coverage_csv),
        "n_models": len(models),
        "layers": [name for name, _ in layers],
    }, indent=2))


if __name__ == "__main__":
    main()
