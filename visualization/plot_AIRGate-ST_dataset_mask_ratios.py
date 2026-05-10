"""AIRGate-ST cross-dataset summary across the 20% / 40% / 60% mask ratios.

Aggregates AIRGate-ST's per-(dataset, ratio) metrics into:

  * a heatmap showing which (dataset, ratio) combinations have results,
  * a per-ratio box-and-scatter chart over datasets,
  * a per-dataset line plot over the three ratios.

Each figure focuses solely on AIRGate-ST so the project's primary model
can be inspected without the noise of the other six methods.
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
METHOD = "AIRGate-ST"
LAYERS = ["all_holdout", "gt_positive"]
METRICS = [
    ("rmse", "RMSE", "lower"),
    ("mae", "MAE", "lower"),
    ("cos", "Cosine", "higher"),
    ("pearson_r", "PCC", "higher"),
]
RATIOS = [20, 40, 60]


def parse_result_dir(name: str):
    """Return ``(sample_id, ratio)`` if the run-directory name matches the expected pattern."""
    if not (re.match(r"151\d{3}_seed\d+_e100_per_site_random_", name) or name.startswith("seed1531_e100_per_site_random_")):
        return None
    ratio_match = re.search(r"per_site_random_(20|40|60)_seed", name)
    if not ratio_match:
        return None
    ratio = int(ratio_match.group(1))
    sample_match = re.match(r"(151\d{3})_", name)
    sample_id = sample_match.group(1) if sample_match else "151673"
    return sample_id, ratio


def read_airgate_rows(root: str):
    """Read every available (sample, ratio) AIRGate-ST result row from ``results/``."""
    rows = []
    results_dir = os.path.join(root, "results")
    for name in os.listdir(results_dir):
        parsed = parse_result_dir(name)
        if parsed is None:
            continue
        sample_id, ratio = parsed
        metrics_path = os.path.join(results_dir, name, "metrics", "layered_metrics.csv")
        if not os.path.exists(metrics_path):
            continue
        with open(metrics_path, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                method = row["method"]
                if method != METHOD or row["layer"] not in LAYERS:
                    continue
                out = {
                    "sample_id": sample_id,
                    "mask_ratio": ratio,
                    "layer": row["layer"],
                    "n": int(row["n"]),
                }
                for metric, _label, _direction in METRICS:
                    out[metric] = float(row[metric])
                rows.append(out)
    return rows


def write_summary(rows, out_dir):
    """Write the long-form per-row summary CSV that backs all three figures."""
    path = os.path.join(out_dir, "AIRGate-ST_dataset_mask_ratio_summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["sample_id", "mask_ratio", "layer", "n"] + [m[0] for m in METRICS]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["sample_id"], r["mask_ratio"], r["layer"])):
            writer.writerow(row)
    return path


def plot_availability(rows, out_dir):
    """Render a (sample x ratio) presence heatmap for AIRGate-ST runs."""
    samples = sorted({row["sample_id"] for row in rows})
    available = {(row["sample_id"], row["mask_ratio"]) for row in rows}
    mat = np.zeros((len(samples), len(RATIOS)), dtype=float)
    for i, sample_id in enumerate(samples):
        for j, ratio in enumerate(RATIOS):
            if (sample_id, ratio) in available:
                mat[i, j] = 1.0

    fig, ax = plt.subplots(figsize=(5.5, max(3.5, 0.35 * len(samples))), constrained_layout=True)
    ax.imshow(mat, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(RATIOS)), [str(r / 100) for r in RATIOS])
    ax.set_yticks(range(len(samples)), samples)
    ax.set_xlabel("Mask Ratio")
    ax.set_title("Available AIRGate-ST metrics by dataset")
    for i in range(len(samples)):
        for j in range(len(RATIOS)):
            ax.text(j, i, "yes" if mat[i, j] else "-", ha="center", va="center", fontsize=8)
    path = os.path.join(out_dir, "AIRGate-ST_dataset_ratio_availability.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_box_scatter(rows, out_dir):
    """Render box + scatter charts of AIRGate-ST metrics across datasets, one per ratio."""
    by_key = defaultdict(list)
    for row in rows:
        by_key[(row["layer"], row["mask_ratio"])].append(row)

    fig, axes = plt.subplots(len(LAYERS), len(METRICS), figsize=(16, 7.5), constrained_layout=True)
    rng = np.random.default_rng(7)
    for row_idx, layer in enumerate(LAYERS):
        for col_idx, (metric, label, _direction) in enumerate(METRICS):
            ax = axes[row_idx][col_idx]
            positions = np.arange(len(RATIOS), dtype=float)
            values_by_ratio = []
            for ratio in RATIOS:
                values_by_ratio.append([row[metric] for row in by_key.get((layer, ratio), [])])

            box_data = [vals if vals else [np.nan] for vals in values_by_ratio]
            ax.boxplot(
                box_data,
                positions=positions,
                widths=0.35,
                showfliers=False,
                patch_artist=True,
                boxprops={"facecolor": "#d9e8fb", "edgecolor": "#4c78a8"},
                medianprops={"color": "#1f4e79", "linewidth": 1.4},
            )
            for pos, ratio, vals in zip(positions, RATIOS, values_by_ratio):
                if not vals:
                    continue
                jitter = rng.normal(0.0, 0.035, size=len(vals))
                ax.scatter(np.full(len(vals), pos) + jitter, vals, s=28, color="#d95f02", alpha=0.85, zorder=3)
                for point_row, x, y in zip(by_key[(layer, ratio)], np.full(len(vals), pos) + jitter, vals):
                    if len(vals) <= 3:
                        ax.text(x + 0.025, y, point_row["sample_id"], fontsize=7, va="center")

            ax.set_title(f"{layer} | {label}")
            ax.set_xticks(positions, [str(r / 100) for r in RATIOS])
            ax.set_xlabel("Mask Ratio")
            ax.set_ylabel("Value")
            ax.grid(True, axis="y", alpha=0.25)
            counts = [len(vals) for vals in values_by_ratio]
            subtitle = " / ".join(f"n@{r/100:g}={c}" for r, c in zip(RATIOS, counts))
            ax.text(0.5, -0.22, subtitle, transform=ax.transAxes, ha="center", va="top", fontsize=8)

    fig.suptitle("AIRGate-ST: dataset distribution across mask ratios", fontsize=13)
    path = os.path.join(out_dir, "AIRGate-ST_dataset_mask_ratio_box_scatter.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_dataset_lines(rows, out_dir):
    """Render per-dataset metric curves over the three ratios, one line per sample."""
    data = defaultdict(dict)
    for row in rows:
        data[(row["sample_id"], row["layer"], row["mask_ratio"])] = row
    samples = sorted({row["sample_id"] for row in rows})

    paths = []
    for layer in LAYERS:
        fig, axes = plt.subplots(1, len(METRICS), figsize=(16, 4.0), constrained_layout=True)
        for ax, (metric, label, _direction) in zip(axes, METRICS):
            for sample_id in samples:
                xs = []
                ys = []
                for ratio in RATIOS:
                    row = data.get((sample_id, layer, ratio))
                    if row is None:
                        continue
                    xs.append(ratio / 100)
                    ys.append(row[metric])
                if not xs:
                    continue
                ax.plot(xs, ys, marker="o", linewidth=1.1, alpha=0.75, label=sample_id)
            ax.set_title(label)
            ax.set_xlabel("Mask Ratio")
            ax.set_ylabel("Value")
            ax.set_xticks([r / 100 for r in RATIOS])
            ax.grid(True, alpha=0.25)
        axes[-1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
        fig.suptitle(f"AIRGate-ST: {layer}", fontsize=13)
        path = os.path.join(out_dir, f"AIRGate-ST_dataset_lines_{layer}.png")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def main():
    """CLI entry point: collect rows, write the summary CSV, render the three figures."""
    parser = argparse.ArgumentParser(description="Plot AIRGate-ST metrics per dataset across mask ratios.")
    parser.add_argument("--root", default=PROJECT_ROOT)
    parser.add_argument("--out_dir", default=os.path.join(PROJECT_ROOT, "results", "visualization", "AIRGate-ST_dataset_mask_ratios"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = read_airgate_rows(args.root)
    paths = [
        write_summary(rows, args.out_dir),
        plot_availability(rows, args.out_dir),
        plot_box_scatter(rows, args.out_dir),
        *plot_dataset_lines(rows, args.out_dir),
    ]
    print(f"[Plot] rows={len(rows)} files={len(paths)} -> {args.out_dir}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
