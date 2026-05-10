"""Plot selected model metrics across the final 20/40/60% mask ratios."""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
METHODS = [
    "Mean Baseline",
    "SoftImpute",
    "Multiple Imputation",
    "Spatial KNN",
    "Spatial IDW",
    "AIRGate-ST",
    "AIRDiff-ST",
]
SHORT = {
    "Mean Baseline": "Mean",
    "SoftImpute": "SoftImpute",
    "Multiple Imputation": "MI",
    "Spatial KNN": "Spatial KNN",
    "Spatial IDW": "Spatial IDW",
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
COLORS = {
    "Mean Baseline": "#333333",
    "SoftImpute": "#999999",
    "Multiple Imputation": "#1b9e77",
    "Spatial KNN": "#7570b3",
    "Spatial IDW": "#e7298a",
    "AIRGate-ST": "#d95f02",
    "AIRDiff-ST": "#1f78b4",
}


def read_data(root: str, ratios: list[int]):
    """Load per-ratio ``layered_metrics.csv`` files for sample 151673.

    Returns ``data[ratio][(method, layer, metric)] = float``. Missing rows
    are skipped silently; the caller's plotting code handles gaps.
    """
    data = {ratio: {} for ratio in ratios}
    for ratio in ratios:
        path = os.path.join(
            root,
            "results",
            f"seed1531_e100_per_site_random_{ratio}_seed1531",
            "metrics",
            "layered_metrics.csv",
        )
        with open(path, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                method = row["method"]
                if method not in METHODS or row["layer"] not in LAYERS:
                    continue
                data[ratio].setdefault(row["layer"], {})[method] = {
                    "n": int(row["n"]),
                    "rmse": float(row["rmse"]),
                    "mae": float(row["mae"]),
                    "cos": float(row["cos"]),
                    "pearson_r": float(row["pearson_r"]),
                }
    return data


def signed_improvement(mean_value: float, model_value: float, direction: str) -> float:
    """Return ``model_value - mean_value`` flipped so positive = better.

    For ``direction == 'lower'`` (RMSE/MAE), better means smaller, so the
    sign is inverted; for ``'higher'`` (cosine/PCC) the raw difference is
    already positive when better.
    """
    if direction == "lower":
        return mean_value - model_value
    return model_value - mean_value


def rank_values(values: dict[str, float], direction: str) -> dict[str, int]:
    """Rank methods 1..N at one (ratio, layer, metric); 1 is best for the given direction."""
    reverse = direction == "higher"
    ordered = sorted(values.items(), key=lambda x: x[1], reverse=reverse)
    return {method: idx + 1 for idx, (method, _value) in enumerate(ordered)}


def plot_rank_heatmap(data, ratios, out_dir):
    """Render a ranks-by-method-and-(layer/metric) heatmap PNG."""
    fig, axes = plt.subplots(len(LAYERS), len(METRICS), figsize=(16, 7.8), constrained_layout=True)
    for row_idx, layer in enumerate(LAYERS):
        for col_idx, (metric, label, direction) in enumerate(METRICS):
            ax = axes[row_idx][col_idx]
            mat = []
            for method in METHODS:
                method_ranks = []
                for ratio in ratios:
                    vals = {m: data[ratio][layer][m][metric] for m in METHODS}
                    method_ranks.append(rank_values(vals, direction)[method])
                mat.append(method_ranks)
            arr = np.asarray(mat, dtype=float)
            im = ax.imshow(arr, cmap="viridis_r", vmin=1, vmax=len(METHODS), aspect="auto")
            ax.set_title(f"{layer} | {label} rank")
            ax.set_xticks(range(len(ratios)), [str(r / 100) for r in ratios])
            ax.set_yticks(range(len(METHODS)), [SHORT[m] for m in METHODS], fontsize=8)
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    ax.text(j, i, str(int(arr[i, j])), ha="center", va="center", fontsize=8, color="white" if arr[i, j] >= 4 else "black")
            fig.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle("Selected models: ranking across mask ratios (1 = best)", fontsize=13)
    path = os.path.join(out_dir, "selected_models_rank_heatmap.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_delta_vs_mean(data, ratios, out_dir):
    """Render ``model_metric - mean_baseline_metric`` curves vs mask ratio."""
    fig, axes = plt.subplots(len(LAYERS), len(METRICS), figsize=(16, 7.8), constrained_layout=True)
    x = [r / 100 for r in ratios]
    compare_methods = [m for m in METHODS if m != "Mean Baseline"]
    for row_idx, layer in enumerate(LAYERS):
        for col_idx, (metric, label, direction) in enumerate(METRICS):
            ax = axes[row_idx][col_idx]
            for method in compare_methods:
                y = []
                for ratio in ratios:
                    mean_value = data[ratio][layer]["Mean Baseline"][metric]
                    model_value = data[ratio][layer][method][metric]
                    y.append(signed_improvement(mean_value, model_value, direction))
                ax.plot(x, y, marker="o", linewidth=1.5, label=SHORT[method], color=COLORS[method])
            ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
            ax.set_title(f"{layer} | {label}\n> 0 beats Mean")
            ax.set_xlabel("Mask Ratio")
            ax.set_ylabel("Signed improvement")
            ax.set_xticks(x)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=7, loc="best")
    fig.suptitle("Selected models: signed delta vs Mean", fontsize=13)
    path = os.path.join(out_dir, "selected_models_delta_vs_mean.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_raw_lines(data, ratios, out_dir):
    """Render raw metric curves vs mask ratio (the line plot used in notebook Section 5)."""
    fig, axes = plt.subplots(len(LAYERS), len(METRICS), figsize=(16, 7.8), constrained_layout=True)
    x = [r / 100 for r in ratios]
    for row_idx, layer in enumerate(LAYERS):
        for col_idx, (metric, label, _direction) in enumerate(METRICS):
            ax = axes[row_idx][col_idx]
            for method in METHODS:
                y = [data[ratio][layer][method][metric] for ratio in ratios]
                lw = 2.2 if method in {"Mean Baseline", "AIRGate-ST"} else 1.2
                alpha = 1.0 if method in {"Mean Baseline", "AIRGate-ST"} else 0.7
                ax.plot(x, y, marker="o", linewidth=lw, alpha=alpha, label=SHORT[method], color=COLORS[method])
            ax.set_title(f"{layer} | {label}")
            ax.set_xlabel("Mask Ratio")
            ax.set_ylabel("Value")
            ax.set_xticks(x)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=7, loc="best")
    fig.suptitle("Selected models: raw metrics", fontsize=13)
    path = os.path.join(out_dir, "selected_models_raw_lines.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def write_summary_csv(data, ratios, out_dir):
    """Write the long-form summary CSV that backs the line and heatmap figures."""
    path = os.path.join(out_dir, "selected_models_mask_ratio_summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "mask_ratio",
            "layer",
            "method",
            "metric",
            "value",
            "rank",
            "signed_improvement_vs_mean",
        ])
        for ratio in ratios:
            for layer in LAYERS:
                for metric, _label, direction in METRICS:
                    vals = {m: data[ratio][layer][m][metric] for m in METHODS}
                    ranks = rank_values(vals, direction)
                    mean_value = vals["Mean Baseline"]
                    for method in METHODS:
                        writer.writerow([
                            ratio,
                            layer,
                            method,
                            metric,
                            vals[method],
                            ranks[method],
                            signed_improvement(mean_value, vals[method], direction),
                        ])
    return path


def main():
    """CLI entry point: load metrics, write the summary CSV, render the three figures."""
    parser = argparse.ArgumentParser(description="Plot selected model comparisons across mask ratios.")
    parser.add_argument("--root", default=PROJECT_ROOT)
    parser.add_argument("--ratios", default="20,40,60")
    parser.add_argument("--out_dir", default=os.path.join(PROJECT_ROOT, "results", "visualization", "selected_models_mask_ratio"))
    args = parser.parse_args()

    ratios = [int(x.strip()) for x in args.ratios.split(",") if x.strip()]
    os.makedirs(args.out_dir, exist_ok=True)
    data = read_data(args.root, ratios)
    paths = [
        plot_rank_heatmap(data, ratios, args.out_dir),
        plot_delta_vs_mean(data, ratios, args.out_dir),
        plot_raw_lines(data, ratios, args.out_dir),
        write_summary_csv(data, ratios, args.out_dir),
    ]
    print(f"[Plot] wrote {len(paths)} files -> {args.out_dir}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
