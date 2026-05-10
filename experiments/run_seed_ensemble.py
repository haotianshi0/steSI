import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from typing import Dict, Iterable, List


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON = sys.executable
METRIC_FIELDS = ["n", "rmse", "mae", "cos", "pearson_r"]


def parse_seeds(value: str) -> List[int]:
    seeds = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer seed")
    return seeds


def run_cmd(cmd: List[str], cwd: str, label: str, dry_run: bool) -> None:
    print(f"[Run] {label}: " + " ".join(f'"{x}"' if " " in x else x for x in cmd), flush=True)
    if dry_run:
        print(f"[DryRun] {label}", flush=True)
        return
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"[Error] {label}: exit_code={exc.returncode}", flush=True)
        raise
    print(f"[Done] {label}", flush=True)


def format_mask_name(mask_name: str, mask_seed: int, model_seed: int) -> str:
    return mask_name.format(seed=mask_seed, mask_seed=mask_seed, model_seed=model_seed)


def run_name_for(sample_id: str, mask_seed: int, model_seed: int, epochs: int, mask_name: str) -> str:
    resolved_mask = format_mask_name(mask_name, mask_seed, model_seed)
    sample_prefix = "" if sample_id == "151673" else f"{sample_id}_"
    return f"{sample_prefix}mask{mask_seed}_model{model_seed}_e{epochs}_{resolved_mask}"


def read_layered_metrics(path: str, mask_seed: int, model_seed: int) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing layered metrics: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["mask_seed"] = str(mask_seed)
        row["model_seed"] = str(model_seed)
    return rows


def to_float(value: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def finite_values(values: Iterable[float]) -> List[float]:
    return [v for v in values if math.isfinite(v)]


def aggregate_rows(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["layer"])].append(row)

    out = []
    for (method, layer), group in sorted(groups.items()):
        item: Dict[str, object] = {
            "method": method,
            "layer": layer,
            "model_seed_count": len({row["model_seed"] for row in group}),
        }
        for field in METRIC_FIELDS:
            vals = finite_values(to_float(row.get(field, "")) for row in group)
            item[f"{field}_mean"] = statistics.fmean(vals) if vals else float("nan")
            item[f"{field}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        out.append(item)
    return out


def write_csv(path: str, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Saved] {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and aggregate multi-seed spatial imputation benchmarks.")
    parser.add_argument("--sample_id", default="151673", help="Dataset/sample id under data/<sample_id>.")
    parser.add_argument("--h5ad", default=None, help="Optional explicit h5ad path.")
    parser.add_argument("--seeds", default=None, help="Legacy alias for --model_seeds.")
    parser.add_argument("--mask_seed", type=int, default=10, help="Fixed seed controlling train/val mask generation.")
    parser.add_argument("--model_seeds", default="6,10,42", help="Comma-separated model seeds, e.g. 6,10,42")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mask_name", default="per_site_random_60_seed{seed}")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--models", default="all", help="Comma-separated model keys to pass to run_harder_benchmark.py.")
    parser.add_argument("--skip_dual_block", action="store_true", help="Skip dual-block heatmap generation in each run.")
    parser.add_argument("--shared_mask_cache", action="store_true", help="Store/reuse masks under results/shared_masks by mask_seed.")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--skip_prediction_ensemble", action="store_true")
    parser.add_argument("--skip_significance", action="store_true")
    parser.add_argument("--bootstrap_iters", type=int, default=1000)
    parser.add_argument("--bootstrap_sample_size", type=int, default=200000)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    model_seeds = parse_seeds(args.seeds or args.model_seeds)
    mask_label = (
        args.mask_name
        .replace("{seed}", str(args.mask_seed))
        .replace("{mask_seed}", str(args.mask_seed))
        .replace("{model_seed}", "MODEL")
    )
    sample_label = "" if args.sample_id == "151673" else f"{args.sample_id}_"
    out_dir = args.out_dir or os.path.join(
        PROJECT_ROOT,
        "results",
        f"{sample_label}prediction_ensemble_mask{args.mask_seed}_models{'_'.join(str(s) for s in model_seeds)}_e{args.epochs}_{mask_label}",
    )
    os.makedirs(out_dir, exist_ok=True)

    seed_runs = []
    for model_seed in model_seeds:
        run_name = run_name_for(args.sample_id, args.mask_seed, model_seed, args.epochs, args.mask_name)
        run_root = os.path.join(PROJECT_ROOT, "results", run_name)
        cmd = [
            PYTHON,
            os.path.join(PROJECT_ROOT, "experiments", "run_harder_benchmark.py"),
            "--sample_id", args.sample_id,
            "--seed", str(model_seed),
            "--mask_seed", str(args.mask_seed),
            "--model_seed", str(model_seed),
            "--epochs", str(args.epochs),
            "--device", args.device,
            "--mask_name", args.mask_name,
            "--run_name", run_name,
            "--models", args.models,
        ]
        if args.h5ad:
            cmd.extend(["--h5ad", args.h5ad])
        if args.skip_dual_block:
            cmd.append("--skip_dual_block")
        if args.shared_mask_cache:
            cmd.append("--shared_mask_cache")
        if args.skip_existing:
            cmd.append("--skip_existing")
        if args.dry_run:
            cmd.append("--dry_run")
        run_cmd(cmd, PROJECT_ROOT, f"benchmark mask_seed={args.mask_seed} model_seed={model_seed}", args.dry_run)
        seed_runs.append({
            "mask_seed": args.mask_seed,
            "model_seed": model_seed,
            "run_name": run_name,
            "run_root": run_root,
        })

    if args.dry_run:
        print(f"[DryRun] would aggregate metrics -> {out_dir}")
        return

    all_rows = []
    for item in seed_runs:
        metrics_path = os.path.join(item["run_root"], "metrics", "layered_metrics.csv")
        all_rows.extend(read_layered_metrics(metrics_path, item["mask_seed"], item["model_seed"]))

    raw_path = os.path.join(out_dir, "model_seed_layered_metrics.csv")
    raw_fieldnames = ["mask_seed", "model_seed", "method", "layer", "n", "rmse", "mae", "cos", "pearson_r", "pred_path", "mask_dir"]
    write_csv(raw_path, all_rows, raw_fieldnames)

    aggregated = aggregate_rows(all_rows)
    aggregate_path = os.path.join(out_dir, "model_seed_metric_summary.csv")
    aggregate_fields = ["method", "layer", "model_seed_count"]
    for field in METRIC_FIELDS:
        aggregate_fields.extend([f"{field}_mean", f"{field}_std"])
    write_csv(aggregate_path, aggregated, aggregate_fields)

    ensemble_registry = None
    if not args.skip_prediction_ensemble:
        cmd = [
            PYTHON,
            os.path.join(PROJECT_ROOT, "experiments", "build_prediction_ensemble.py"),
            "--mask_seed", str(args.mask_seed),
            "--model_seeds", ",".join(str(s) for s in model_seeds),
            "--epochs", str(args.epochs),
            "--mask_name", args.mask_name,
            "--sample_id", args.sample_id,
            "--out_dir", os.path.join(out_dir, "prediction_ensemble"),
        ]
        run_cmd(cmd, PROJECT_ROOT, "prediction-level ensemble", args.dry_run)
        ensemble_registry = os.path.join(out_dir, "prediction_ensemble", "model_registry.csv")

    significance_csv = None
    if not args.skip_significance and ensemble_registry is not None:
        first_run = seed_runs[0]
        mask_dir = os.path.join(
            first_run["run_root"],
            f"masks_{args.sample_id}_seed{args.mask_seed}",
            format_mask_name(args.mask_name, args.mask_seed, first_run["model_seed"]),
        )
        significance_out = os.path.join(out_dir, "significance")
        cmd = [
            PYTHON,
            os.path.join(PROJECT_ROOT, "evaluation", "paired_significance.py"),
            "--mask_dir", mask_dir,
            "--model_registry", ensemble_registry,
            "--out_dir", significance_out,
            "--baseline", "Mean Baseline Ensemble",
            "--bootstrap_iters", str(args.bootstrap_iters),
            "--bootstrap_sample_size", str(args.bootstrap_sample_size),
        ]
        run_cmd(cmd, PROJECT_ROOT, "paired significance", args.dry_run)
        significance_csv = os.path.join(significance_out, "paired_significance.csv")

    summary = {
        "mask_seed": args.mask_seed,
        "sample_id": args.sample_id,
        "model_seeds": model_seeds,
        "epochs": args.epochs,
        "device": args.device,
        "mask_name": args.mask_name,
        "seed_runs": seed_runs,
        "raw_metrics_csv": os.path.abspath(raw_path),
        "aggregate_metrics_csv": os.path.abspath(aggregate_path),
        "prediction_ensemble_registry": os.path.abspath(ensemble_registry) if ensemble_registry else None,
        "significance_csv": os.path.abspath(significance_csv) if significance_csv else None,
    }
    summary_path = os.path.join(out_dir, "seed_ensemble_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[Saved] {summary_path}")
    print(f"[Done] seed ensemble -> {out_dir}")


if __name__ == "__main__":
    main()
