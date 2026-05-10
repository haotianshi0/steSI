"""
3x3 method-comparison heatmap on a high-signal observed-only block.

Layout:
    [GT]          [Masked Train]   [Mean]
    [SoftImpute]  [MI]             [Spatial KNN]
    [Spatial IDW] [AIRGate-ST]     [AIRDiff-ST]

The block (rows x sites) is selected once via ``select_block`` with mode
``high_signal`` so that all 9 panels share the same sub-matrix and the same
column ordering (descending mean GT ratio).  Each model's full prediction
matrix is sliced with the same ``rows`` and ``cols``, and panels share a
single magma colorbar with NaN rendered as light gray.

Inputs read from ``--run_dir``:
    * ``model_registry.csv``                -- maps method name to pred_ratio.npy
    * ``<baselines_subdir>/gt_ratio.npy``   -- ground-truth ratio matrix
    * ``<baselines_subdir>/train_ratio.npy``-- training-visible ratio matrix
    * ``<baselines_subdir>/val_mask.npy``   -- holdout mask used to select the block

Predictions for non-core methods present in the registry are silently ignored.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "visualization"))

from dual_block_observed_heatmap import select_block, _mean_column_order  # noqa: E402


CORE_METHODS: List[str] = [
    "Mean Baseline",
    "SoftImpute",
    "Multiple Imputation",
    "Spatial KNN",
    "Spatial IDW",
    "AIRGate-ST",
    "AIRDiff-ST",
]
SHORT_LABEL: Dict[str, str] = {
    "Mean Baseline": "Mean",
    "SoftImpute": "SoftImpute",
    "Multiple Imputation": "MI",
    "Spatial KNN": "Spatial KNN",
    "Spatial IDW": "Spatial IDW",
    "AIRGate-ST": "AIRGate-ST",
    "AIRDiff-ST": "AIRDiff-ST",
}


def _resolve(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(REPO_ROOT, path))


def read_registry(run_dir: str) -> Dict[str, str]:
    registry_path = os.path.join(run_dir, "model_registry.csv")
    if not os.path.exists(registry_path):
        raise FileNotFoundError(f"model_registry.csv not found under {run_dir}")
    methods: Dict[str, str] = {}
    with open(registry_path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            method = row["method"]
            if method in CORE_METHODS and method not in methods:
                methods[method] = row["pred_path"]
    missing = [m for m in CORE_METHODS if m not in methods]
    if missing:
        raise RuntimeError(
            f"Registry under {run_dir} is missing core methods: {missing}"
        )
    return methods


def load_shared_arrays(baselines_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gt = np.load(os.path.join(baselines_dir, "gt_ratio.npy")).astype(np.float32)
    train = np.load(os.path.join(baselines_dir, "train_ratio.npy")).astype(np.float32)
    val_mask = np.load(os.path.join(baselines_dir, "val_mask.npy")).astype(bool)
    return gt, train, val_mask


def load_pred(method: str, pred_path: str, expected_shape: Tuple[int, int]) -> np.ndarray:
    arr = np.load(_resolve(pred_path)).astype(np.float32)
    if arr.shape != expected_shape:
        raise ValueError(
            f"{method} pred shape {arr.shape} differs from GT shape {expected_shape}"
        )
    return arr


def draw_3x3_panels(
    panels: List[Tuple[str, np.ndarray]],
    col_order: np.ndarray,
    save_path: str,
    suptitle: str = "",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    if len(panels) != 9:
        raise ValueError(f"Expected 9 panels, got {len(panels)}")

    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("lightgray")

    fig, axes = plt.subplots(3, 3, figsize=(12, 21))
    im = None
    for ax, (title, arr) in zip(axes.flat, panels):
        ordered = arr[:, col_order]
        im = ax.imshow(
            ordered,
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.subplots_adjust(left=0.03, right=0.90, top=0.96, bottom=0.02,
                        wspace=0.08, hspace=0.10)
    if im is not None:
        cax = fig.add_axes([0.92, 0.08, 0.015, 0.84])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("A-to-I editing ratio")

    if suptitle:
        fig.suptitle(suptitle, fontsize=12, y=0.99)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] saved -> {save_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="3x3 method-comparison heatmap on a high-signal observed-only block."
    )
    ap.add_argument(
        "--run_dir",
        required=True,
        help="Run directory containing model_registry.csv and the baselines subdir.",
    )
    ap.add_argument(
        "--baselines_subdir",
        default="baselines",
        help="Subdirectory under --run_dir holding gt_ratio.npy / train_ratio.npy / val_mask.npy.",
    )
    ap.add_argument(
        "--out_path",
        required=True,
        help="Path to write the 3x3 PNG.",
    )
    ap.add_argument("--top_k_sites", type=int, default=50)
    ap.add_argument("--high_signal_min_spots", type=int, default=32)
    ap.add_argument("--high_signal_max_spots", type=int, default=64)
    ap.add_argument("--high_signal_threshold", type=float, default=0.05)
    ap.add_argument(
        "--block_label",
        default="",
        help="Suptitle label describing dataset, mask ratio, and block.",
    )
    args = ap.parse_args()

    run_dir = _resolve(args.run_dir)
    out_path = _resolve(args.out_path)
    baselines_dir = os.path.join(run_dir, args.baselines_subdir)

    print(f"[Run] using run_dir = {run_dir}")
    gt_ratio, train_ratio, val_mask = load_shared_arrays(baselines_dir)
    methods = read_registry(run_dir)

    high_signal = select_block(
        gt_ratio=gt_ratio,
        val_mask=val_mask,
        top_k_sites=args.top_k_sites,
        min_spots=args.high_signal_min_spots,
        max_spots=args.high_signal_max_spots,
        mode="high_signal",
        high_signal_threshold=args.high_signal_threshold,
    )
    rows = high_signal["rows"]
    cols = high_signal["cols"]
    print(f"[Block] selected {len(rows)} spots x {len(cols)} sites")

    gt_sub = gt_ratio[np.ix_(rows, cols)]
    train_sub = train_ratio[np.ix_(rows, cols)]
    col_idx = _mean_column_order(gt_sub)

    panels: List[Tuple[str, np.ndarray]] = [
        ("GT", gt_sub),
        ("Masked Train (gray = held out)", train_sub),
    ]
    for method in CORE_METHODS:
        pred = load_pred(method, methods[method], gt_ratio.shape)
        panels.append((SHORT_LABEL[method], pred[np.ix_(rows, cols)]))

    block_descriptor = (
        f"high_signal_ge_{args.high_signal_threshold:g} block | "
        f"block: {len(rows)} spots x {len(cols)} sites | "
        f"columns sorted by descending mean GT"
    )
    suptitle = f"{args.block_label} | {block_descriptor}" if args.block_label else block_descriptor

    draw_3x3_panels(panels=panels, col_order=col_idx,
                    save_path=out_path, suptitle=suptitle)

    print("[Done]")


if __name__ == "__main__":
    main()
