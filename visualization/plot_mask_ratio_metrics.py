import argparse
import csv
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_LAYERS = ["all_holdout", "gt_positive"]
METRICS = [
    ("mae", "MAE", "#e41a1c", "o"),
    ("rmse", "RMSE", "#1b7f3a", "o"),
    ("pearson_r", "PCC", "#1f4e99", "D"),
    ("cos", "Cosine", "#7a1f7a", "D"),
]


def safe_name(value: str) -> str:
    value = value.lower()
    value = re.sub(r"\(.*?\)", "", value)
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value or "method"


def read_metrics(root: str, ratios: list[int]) -> dict[str, dict[str, dict[int, dict[str, float]]]]:
    data = defaultdict(lambda: defaultdict(dict))
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
                data[row["method"]][row["layer"]][ratio] = {
                    "n": float(row["n"]),
                    "rmse": float(row["rmse"]),
                    "mae": float(row["mae"]),
                    "cos": float(row["cos"]),
                    "pearson_r": float(row["pearson_r"]),
                }
    return data


def plot_method(method: str,
                method_data: dict[str, dict[int, dict[str, float]]],
                ratios: list[int],
                layers: list[str],
                out_dir: str) -> str:
    n_layers = len(layers)
    fig, axes = plt.subplots(1, n_layers, figsize=(5.0 * n_layers, 4.2), constrained_layout=True)
    if n_layers == 1:
        axes = [axes]
    else:
        axes = list(axes)

    for ax, layer in zip(axes, layers):
        layer_data = method_data.get(layer, {})
        for key, label, color, marker in METRICS:
            y = [layer_data.get(ratio, {}).get(key, float("nan")) for ratio in ratios]
            ax.plot(
                [ratio / 100 for ratio in ratios],
                y,
                color=color,
                marker=marker,
                linewidth=1.6,
                markersize=4.8,
                alpha=0.9,
                label=label,
            )

        n_values = [int(layer_data[ratio]["n"]) for ratio in ratios if ratio in layer_data]
        n_label = f" | n={n_values[0]:,}-{n_values[-1]:,}" if n_values else ""
        ax.set_title(f"{layer}{n_label}", fontsize=10)
        ax.set_xlabel("Mask Ratio")
        ax.set_ylabel("Value")
        ax.set_xticks([ratio / 100 for ratio in ratios])
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(method, fontsize=12)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{safe_name(method)}_mask_ratio_metrics.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot metrics vs mask ratio, one figure per method.")
    parser.add_argument("--root", default=PROJECT_ROOT)
    parser.add_argument("--out_dir", default=os.path.join(PROJECT_ROOT, "results", "visualization", "mask_ratio_metrics"))
    parser.add_argument("--ratios", default="20,40,60")
    parser.add_argument("--layers", default=",".join(DEFAULT_LAYERS))
    args = parser.parse_args()

    ratios = [int(x.strip()) for x in args.ratios.split(",") if x.strip()]
    layers = [x.strip() for x in args.layers.split(",") if x.strip()]
    data = read_metrics(args.root, ratios)

    paths = []
    for method in sorted(data):
        paths.append(plot_method(method, data[method], ratios, layers, args.out_dir))

    print(f"[Plot] wrote {len(paths)} figures -> {args.out_dir}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
