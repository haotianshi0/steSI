"""Method-comparison boxplot across the available datasets at one mask ratio.

Scans ``results/`` for run directories whose name encodes the requested
mask-ratio token (e.g. ``per_site_random_60_seed*``), reads each run's
``metrics/layered_metrics.csv``, restricts to the seven core methods, and
draws a 2 x 4 grid of boxplots: rows are the ``all_holdout`` and
``gt_positive`` evaluation layers; columns are RMSE, MAE, cosine, PCC.
Also writes a long-form ``method_comparison_boxplot_summary.csv`` that
contains the underlying per-(sample, method, layer) numbers.
"""

import argparse
import csv
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CORE_METHODS = [
    "Mean Baseline",
    "SoftImpute",
    "Multiple Imputation",
    "Spatial KNN",
    "Spatial IDW",
    "AIRGate-ST",
    "AIRDiff-ST",
]
ALL_METHODS = CORE_METHODS
SHORT = {
    "Mean Baseline": "Mean",
    "SoftImpute": "SoftImpute",
    "Multiple Imputation": "MI",
    "Spatial KNN": "KNN",
    "Spatial IDW": "IDW",
    "AIRGate-ST": "AIRGate-ST",
    "AIRDiff-ST": "AIRDiff-ST",
}
LAYERS = ["all_holdout", "gt_positive"]
METRICS = [
    ("rmse", "RMSE", "lower"),
    ("mae", "MAE", "lower"),
    ("cos", "Cosine", "higher"),
    ("pearson_r", "PCC", "higher"),
]


def parse_result_dir(name: str, ratio: int):
    """Return the sample id encoded in a run-directory name, or ``None`` if it does not match the ratio."""
    ratio_token = f"per_site_random_{ratio}_seed"
    if ratio_token not in name:
        return None
    if name.startswith("seed1531_e100_"):
        return "151673"
    match = re.match(r"(151\d{3})_seed\d+_e100_per_site_random_", name)
    if match:
        return match.group(1)
    return None


def read_rows(root: str, ratio: int, methods: list[str], layers: list[str]) -> list[dict]:
    """Collect one row per (sample, method, layer) at the requested mask ratio.

    Iterates the run directories under ``results/`` and, for every method in
    ``methods`` whose layer is in ``layers``, returns its scalar metrics from
    the per-run ``layered_metrics.csv``.
    """
    results_dir = os.path.join(root, "results")
    by_key = {}
    for name in sorted(os.listdir(results_dir)):
        sample_id = parse_result_dir(name, ratio)
        if sample_id is None:
            continue
        metrics_path = os.path.join(results_dir, name, "metrics", "layered_metrics.csv")
        if not os.path.exists(metrics_path):
            continue
        with open(metrics_path, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                method = row["method"]
                if method not in methods or row["layer"] not in layers:
                    continue
                item = {
                    "sample_id": sample_id,
                    "result_dir": name,
                    "mask_ratio": ratio,
                    "method": method,
                    "layer": row["layer"],
                    "n": int(row["n"]),
                }
                for metric, _label, _direction in METRICS:
                    item[metric] = float(row[metric])
                by_key[(sample_id, method, row["layer"])] = item
    return list(by_key.values())


def write_summary(rows: list[dict], out_dir: str) -> str:
    """Write the long-form per-row summary CSV that backs the boxplot."""
    path = os.path.join(out_dir, "method_comparison_boxplot_summary.csv")
    fieldnames = ["sample_id", "result_dir", "mask_ratio", "method", "layer", "n"] + [m[0] for m in METRICS]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["layer"], r["method"], r["sample_id"])):
            writer.writerow(row)
    return path


def values_by_method(rows: list[dict], layer: str, metric: str, methods: list[str]) -> list[list[float]]:
    """Group ``rows`` into one list of per-sample metric values per method, in ``methods`` order."""
    grouped = defaultdict(list)
    for row in rows:
        if row["layer"] == layer:
            grouped[row["method"]].append(row[metric])
    return [grouped[method] for method in methods]


def plot_boxplots(rows: list[dict], methods: list[str], layers: list[str], out_dir: str, ratio: int) -> str:
    """Render the 2 x 4 boxplot grid (layers x metrics) and return its PNG path.

    AIRGate-ST and AIRDiff-ST boxes are highlighted in orange; baselines
    appear in light blue. Individual sample points are jittered onto each
    box so the underlying spread is visible.
    """
    fig, axes = plt.subplots(len(layers), len(METRICS), figsize=(4.2 * len(METRICS), 4.2 * len(layers)), constrained_layout=True)
    if len(layers) == 1:
        axes = np.asarray([axes])
    rng = np.random.default_rng(19)
    x = np.arange(len(methods), dtype=float)

    for row_idx, layer in enumerate(layers):
        for col_idx, (metric, label, _direction) in enumerate(METRICS):
            ax = axes[row_idx][col_idx]
            data = values_by_method(rows, layer, metric, methods)
            box_data = [vals if vals else [np.nan] for vals in data]
            bp = ax.boxplot(
                box_data,
                positions=x,
                widths=0.55,
                showfliers=False,
                patch_artist=True,
                medianprops={"color": "#111111", "linewidth": 1.4},
                boxprops={"facecolor": "#d7e8f7", "edgecolor": "#4b78a8"},
                whiskerprops={"color": "#4b78a8"},
                capprops={"color": "#4b78a8"},
            )
            for patch, method in zip(bp["boxes"], methods):
                if method == "Mean Baseline":
                    patch.set_facecolor("#e6e6e6")
                elif method == "AIRGate-ST":
                    patch.set_facecolor("#f6c391")
                elif method == "AIRDiff-ST":
                    patch.set_facecolor("#a9d5f2")

            for pos, vals in zip(x, data):
                if not vals:
                    continue
                jitter = rng.normal(0.0, 0.045, size=len(vals))
                ax.scatter(np.full(len(vals), pos) + jitter, vals, s=22, color="#333333", alpha=0.62, zorder=3)

            ax.set_title(f"{layer} | {label}", fontsize=10)
            ax.set_xticks(x, [SHORT[m] for m in methods], rotation=35, ha="right")
            ax.set_ylabel("Value")
            ax.grid(True, axis="y", alpha=0.25)

    sample_count = len({row["sample_id"] for row in rows})
    fig.suptitle(
        f"Method comparison boxplots across {sample_count} datasets, mask ratio={ratio / 100:g}",
        fontsize=13,
    )
    path = os.path.join(out_dir, f"method_comparison_boxplots_ratio_{ratio}.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    """CLI entry point: collect rows, write the summary CSV, render the boxplot PNG."""
    parser = argparse.ArgumentParser(description="Compare methods with boxplots across datasets for one mask ratio.")
    parser.add_argument("--root", default=PROJECT_ROOT)
    parser.add_argument("--ratio", type=int, default=60)
    parser.add_argument("--method_set", choices=["core", "all"], default="core")
    parser.add_argument("--layers", default=",".join(LAYERS))
    parser.add_argument("--out_dir", default=os.path.join(PROJECT_ROOT, "results", "visualization", "method_comparison_boxplots"))
    args = parser.parse_args()

    methods = CORE_METHODS if args.method_set == "core" else ALL_METHODS
    layers = [x.strip() for x in args.layers.split(",") if x.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    rows = read_rows(args.root, args.ratio, methods, layers)
    paths = [
        write_summary(rows, args.out_dir),
        plot_boxplots(rows, methods, layers, args.out_dir, args.ratio),
    ]
    print(f"[Plot] rows={len(rows)} files={len(paths)} -> {args.out_dir}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
