"""Render observed-only block heatmaps for every method in a run's registry.

Driver for ``dual_block_observed_heatmap.py``: reads a run-specific
``model_registry.csv``, selects three sub-blocks (dense, rare-peak,
high-signal) once from the shared ground-truth, and renders the per-model
1x3 + 1x2 figures inside per-method subdirectories. Writes a single
``all_models_dual_block_summary.json`` listing every figure that was
produced.
"""

import argparse
import csv
import json
import os
import re
import sys
from typing import Dict, List

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "visualization"))

from dual_block_observed_heatmap import render_block, select_block  # noqa: E402


def _resolve_path(path: str) -> str:
    """Return ``path`` unchanged if absolute, else resolve relative to the project root."""
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_ROOT, path))


def _safe_key(name: str) -> str:
    """Sanitise a method label into a filesystem-safe lowercase identifier."""
    key = name.lower()
    key = key.replace("baseline", "")
    key = re.sub(r"\(.*?\)", "", key)
    key = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
    return key or "model"


def read_registry(path: str) -> List[Dict[str, str]]:
    """Load a ``method,pred_path`` registry CSV; relative paths are resolved against the project root."""
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({"method": row["method"], "pred_path": _resolve_path(row["pred_path"])})
    return rows


def _load_if_shape(path: str, shape) -> np.ndarray | None:
    """Load ``path`` only if it exists and matches the requested shape; else return ``None``."""
    if not os.path.exists(path):
        return None
    arr = np.load(path)
    if arr.shape != shape:
        return None
    return arr


def load_context(pred_path: str, mask_dir: str | None) -> Dict[str, np.ndarray]:
    """Load the prediction plus the matching gt/train/val arrays needed for rendering.

    Prefers ``mask_dir`` when supplied (so all methods render against the same
    shared reference) and otherwise falls back to the prediction's own
    directory. Raises ``FileNotFoundError`` if any of the four context arrays
    is missing or shape-incompatible.
    """
    pred = np.load(pred_path).astype(np.float32)
    model_dir = os.path.dirname(pred_path)

    # mask_dir is the authoritative shared reference; always prefer it when provided.
    # Fall back to model_dir only when mask_dir is absent.
    if mask_dir:
        gt_ratio = _load_if_shape(os.path.join(mask_dir, "gt_ratio.npy"), pred.shape)
        train_ratio = _load_if_shape(os.path.join(mask_dir, "train_ratio.npy"), pred.shape)
        val_mask = _load_if_shape(os.path.join(mask_dir, "val_mask.npy"), pred.shape)
    else:
        gt_ratio = _load_if_shape(os.path.join(model_dir, "gt_ratio.npy"), pred.shape)
        train_ratio = _load_if_shape(os.path.join(model_dir, "train_ratio.npy"), pred.shape)
        val_mask = _load_if_shape(os.path.join(model_dir, "val_mask.npy"), pred.shape)

    missing = [
        name for name, arr in [
            ("gt_ratio", gt_ratio),
            ("train_ratio", train_ratio),
            ("val_mask", val_mask),
        ] if arr is None
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing compatible context arrays for {pred_path}: {missing}. "
            "Provide --mask_dir containing gt_ratio.npy, train_ratio.npy, val_mask.npy."
        )

    return {
        "pred_ratio": pred,
        "gt_ratio": gt_ratio.astype(np.float32),
        "train_ratio": train_ratio.astype(np.float32),
        "val_mask": val_mask.astype(bool),
    }


def render_model(method: str,
                 pred_path: str,
                 out_dir: str,
                 mask_dir: str | None,
                 top_k_sites: int,
                 dense_min_spots: int,
                 dense_max_spots: int,
                 peak_min_spots: int,
                 peak_max_spots: int,
                 high_signal_min_spots: int,
                 high_signal_max_spots: int,
                 high_signal_threshold: float) -> Dict[str, object]:
    """Render the dense, rare-peak, and high-signal block triplets for one method."""
    ctx = load_context(pred_path, mask_dir)

    dense = select_block(
        gt_ratio=ctx["gt_ratio"],
        val_mask=ctx["val_mask"],
        top_k_sites=top_k_sites,
        min_spots=dense_min_spots,
        max_spots=dense_max_spots,
        mode="dense",
        high_signal_threshold=high_signal_threshold,
    )
    peak = select_block(
        gt_ratio=ctx["gt_ratio"],
        val_mask=ctx["val_mask"],
        top_k_sites=top_k_sites,
        min_spots=peak_min_spots,
        max_spots=peak_max_spots,
        mode="peak",
        high_signal_threshold=high_signal_threshold,
    )
    high_signal = select_block(
        gt_ratio=ctx["gt_ratio"],
        val_mask=ctx["val_mask"],
        top_k_sites=top_k_sites,
        min_spots=high_signal_min_spots,
        max_spots=high_signal_max_spots,
        mode="high_signal",
        high_signal_threshold=high_signal_threshold,
    )

    dense_info = render_block(
        name="dense_block",
        out_dir=out_dir,
        gt_ratio=ctx["gt_ratio"],
        train_ratio=ctx["train_ratio"],
        pred_ratio=ctx["pred_ratio"],
        val_mask=ctx["val_mask"],
        rows=dense["rows"],
        cols=dense["cols"],
    )
    peak_info = render_block(
        name="rare_peak_block",
        out_dir=out_dir,
        gt_ratio=ctx["gt_ratio"],
        train_ratio=ctx["train_ratio"],
        pred_ratio=ctx["pred_ratio"],
        val_mask=ctx["val_mask"],
        rows=peak["rows"],
        cols=peak["cols"],
    )
    high_signal_info = render_block(
        name=f"high_signal_ge_{high_signal_threshold:g}_block",
        out_dir=out_dir,
        gt_ratio=ctx["gt_ratio"],
        train_ratio=ctx["train_ratio"],
        pred_ratio=ctx["pred_ratio"],
        val_mask=ctx["val_mask"],
        rows=high_signal["rows"],
        cols=high_signal["cols"],
    )

    return {
        "method": method,
        "pred_path": os.path.abspath(pred_path),
        "out_dir": os.path.abspath(out_dir),
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
            "threshold": float(high_signal_threshold),
            "selected_n_spots": int(high_signal["selected_n_spots"][0]),
            "available_fully_observed_cols": int(high_signal["fully_observed_cols"][0]),
            **high_signal_info,
        },
    }


def main() -> None:
    """CLI entry point: render observed-only block heatmaps for every method in a registry."""
    ap = argparse.ArgumentParser(description="Generate dual-block observed-only heatmaps for all registry models.")
    ap.add_argument("--model_registry", required=True, help="Run-specific CSV with columns method,pred_path.")
    ap.add_argument("--mask_dir", default=None, help="Optional shared context directory with gt_ratio.npy, train_ratio.npy, val_mask.npy.")
    ap.add_argument("--out_dir", default=os.path.join(PROJECT_ROOT, "results", "visualization", "all_models_dual_block"))
    ap.add_argument("--top_k_sites", type=int, default=50)
    ap.add_argument("--dense_min_spots", type=int, default=128)
    ap.add_argument("--dense_max_spots", type=int, default=1024)
    ap.add_argument("--peak_min_spots", type=int, default=32)
    ap.add_argument("--peak_max_spots", type=int, default=256)
    ap.add_argument("--high_signal_min_spots", type=int, default=32)
    ap.add_argument("--high_signal_max_spots", type=int, default=64)
    ap.add_argument("--high_signal_threshold", type=float, default=0.05)
    args = ap.parse_args()

    registry_path = _resolve_path(args.model_registry)
    mask_dir = _resolve_path(args.mask_dir) if args.mask_dir else None
    out_root = _resolve_path(args.out_dir)
    os.makedirs(out_root, exist_ok=True)

    summary = {
        "model_registry": registry_path,
        "mask_dir": mask_dir,
        "out_dir": out_root,
        "top_k_sites": int(args.top_k_sites),
        "high_signal_threshold": float(args.high_signal_threshold),
        "methods": {},
    }

    for row in read_registry(registry_path):
        method = row["method"]
        key = _safe_key(method)
        model_out_dir = os.path.join(out_root, key)
        os.makedirs(model_out_dir, exist_ok=True)
        print(f"[DualBlock] {method} -> {model_out_dir}")
        model_summary = render_model(
            method=method,
            pred_path=row["pred_path"],
            out_dir=model_out_dir,
            mask_dir=mask_dir,
            top_k_sites=args.top_k_sites,
            dense_min_spots=args.dense_min_spots,
            dense_max_spots=args.dense_max_spots,
            peak_min_spots=args.peak_min_spots,
            peak_max_spots=args.peak_max_spots,
            high_signal_min_spots=args.high_signal_min_spots,
            high_signal_max_spots=args.high_signal_max_spots,
            high_signal_threshold=args.high_signal_threshold,
        )
        summary["methods"][key] = model_summary
        with open(os.path.join(model_out_dir, "dual_block_summary.json"), "w", encoding="utf-8") as f:
            json.dump(model_summary, f, indent=2, ensure_ascii=False)

    summary_path = os.path.join(out_root, "all_models_dual_block_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps({"summary_path": summary_path, "n_models": len(summary["methods"])}, indent=2))


if __name__ == "__main__":
    main()
