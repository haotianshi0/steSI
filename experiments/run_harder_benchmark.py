"""Run the full imputation benchmark for one dataset, mask, and seed pair.

The script creates or reuses a shared train/holdout mask, runs the selected
models, writes a model registry, computes layered metrics, and optionally
generates dual-block heatmaps for qualitative inspection.
"""

import argparse
import csv
import os
import subprocess
import sys
from collections import OrderedDict
from typing import List

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON = sys.executable


def run_cmd(cmd: List[str], cwd: str, dry_run: bool = False, label: str | None = None) -> None:
    name = label or os.path.basename(cmd[1]) if len(cmd) > 1 else "command"
    print(f"[Run] {name}: " + " ".join(f'"{x}"' if " " in x else x for x in cmd), flush=True)
    if dry_run:
        print(f"[DryRun] {name}", flush=True)
        return
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"[Error] {name}: exit_code={exc.returncode}", flush=True)
        raise
    print(f"[Done] {name}", flush=True)


def format_mask_name(mask_name: str, mask_seed: int, model_seed: int) -> str:
    return mask_name.format(seed=mask_seed, mask_seed=mask_seed, model_seed=model_seed)


def default_run_name(sample_id: str, mask_seed: int, model_seed: int, legacy_seed: int, epochs: int, resolved_mask_name: str) -> str:
    sample_prefix = "" if sample_id == "151673" else f"{sample_id}_"
    if mask_seed == model_seed == legacy_seed:
        return f"{sample_prefix}seed{legacy_seed}_e{epochs}_{resolved_mask_name}"
    return f"{sample_prefix}mask{mask_seed}_model{model_seed}_e{epochs}_{resolved_mask_name}"


def ensure_masks(args, run_root: str) -> str:
    masks_root = (
        os.path.join(PROJECT_ROOT, "results", "shared_masks", args.sample_id, f"masks_seed{args.mask_seed}")
        if args.shared_mask_cache
        else os.path.join(run_root, f"masks_{args.sample_id}_seed{args.mask_seed}")
    )
    mask_dir = os.path.join(masks_root, format_mask_name(args.mask_name, args.mask_seed, args.model_seed))
    if os.path.exists(os.path.join(mask_dir, "train_mask.npy")) and os.path.exists(os.path.join(mask_dir, "val_mask.npy")):
        print(f"[Mask] using existing {mask_dir}")
        return mask_dir
    run_cmd([
        PYTHON,
        os.path.join(PROJECT_ROOT, "src", "masks", "generate_masks.py"),
        "--seed", str(args.mask_seed),
        "--h5ad", args.h5ad,
        "--out_dir", masks_root,
    ], cwd=PROJECT_ROOT, dry_run=args.dry_run)
    return mask_dir


def parse_models(value: str) -> set[str]:
    aliases = {
        "all": "all",
        "mean": "mean",
        "mean_baseline": "mean",
        "softimpute": "softimpute",
        "multiple_imputation": "multiple_imputation",
        "mi": "multiple_imputation",
        "spatial_knn": "spatial_knn",
        "knn": "spatial_knn",
        "spatial_idw": "spatial_idw",
        "idw": "spatial_idw",
        "airgate_st": "AIRGate-ST",
        "airgate": "AIRGate-ST",
        "airdiff_st": "AIRDiff-ST",
        "airdiff": "AIRDiff-ST",
    }
    all_models = {
        "mean",
        "softimpute",
        "multiple_imputation",
        "spatial_knn",
        "spatial_idw",
        "AIRGate-ST",
        "AIRDiff-ST",
    }
    selected = set()
    for raw in value.split(","):
        key = raw.strip().lower().replace("-", "_").replace(" ", "_")
        if not key:
            continue
        if key not in aliases:
            raise ValueError(f"Unknown model key in --models: {raw}")
        mapped = aliases[key]
        if mapped == "all":
            return all_models
        selected.add(mapped)
    if not selected:
        raise ValueError("--models must include at least one model key")
    return selected


def write_registry(path: str, rows: List[tuple[str, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "pred_path"])
        writer.writerows(rows)
    print(f"[Registry] {path}")


def prediction_file_valid(pred_path: str, mask_dir: str) -> bool:
    if not os.path.exists(pred_path):
        return False
    try:
        pred = np.load(pred_path)
        gt = np.load(os.path.join(mask_dir, "gt_ratio.npy"))
        train_mask = np.load(os.path.join(mask_dir, "train_mask.npy")).astype(bool)
        val_mask = np.load(os.path.join(mask_dir, "val_mask.npy")).astype(bool)
        if pred.shape != gt.shape:
            return False
        pred_finite = np.isfinite(pred)
        if np.any((train_mask | val_mask) & ~pred_finite):
            return False
        finite_pred = pred[pred_finite]
        if finite_pred.size and np.any((finite_pred < 0.0) | (finite_pred > 1.0)):
            return False
        if not np.allclose(pred[train_mask], gt[train_mask], equal_nan=False):
            return False
        return True
    except Exception as exc:
        print(f"[SkipCheck] invalid existing prediction {pred_path}: {exc}")
        return False


def configure_cpu_threads(cpu_threads: int) -> int:
    workers = cpu_threads if cpu_threads > 0 else max(1, os.cpu_count() or 1)
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[key] = str(workers)
    return workers


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the 7-model spatial imputation benchmark with one shared external mask.")
    ap.add_argument("--sample_id", default="151673", help="Dataset/sample id under data/<sample_id>, e.g. 151507.")
    ap.add_argument("--h5ad", default=None, help="Optional explicit h5ad path. Defaults to data/{sample_id}/adata_ai_compressed.h5ad.")
    ap.add_argument("--seed", type=int, default=6, help="Legacy seed used for both mask/model unless overridden.")
    ap.add_argument("--mask_seed", type=int, default=None, help="Seed controlling train/val mask generation.")
    ap.add_argument("--model_seed", type=int, default=None, help="Seed controlling model initialization/training randomness.")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--mask_name",
        default="per_site_random_60_seed{seed}",
        help="Mask subdir name, e.g. per_site_random_60_seed{seed}. Supports {seed}, {mask_seed}, {model_seed}.",
    )
    ap.add_argument("--run_name", default=None, help="Default keeps legacy seed format when mask/model seeds are equal; otherwise uses mask{mask_seed}_model{model_seed}.")
    ap.add_argument("--models", default="all", help="Comma-separated model keys to run/register, or all.")
    ap.add_argument("--skip_dual_block", action="store_true", help="Skip dual-block heatmap generation.")
    ap.add_argument("--shared_mask_cache", action="store_true", help="Store/reuse masks under results/shared_masks by mask_seed.")
    ap.add_argument("--cpu_threads", type=int, default=0, help="CPU threads for NumPy/Sklearn/spatial baselines. 0 means all logical cores.")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--skip_existing", action="store_true", help="Skip model commands when the expected pred file exists.")
    args = ap.parse_args()
    cpu_threads = configure_cpu_threads(args.cpu_threads)
    args.mask_seed = args.seed if args.mask_seed is None else args.mask_seed
    args.model_seed = args.seed if args.model_seed is None else args.model_seed
    args.h5ad = args.h5ad or os.path.join(PROJECT_ROOT, "data", args.sample_id, "adata_ai_compressed.h5ad")
    selected_models = parse_models(args.models)
    if not os.path.exists(args.h5ad):
        raise FileNotFoundError(f"h5ad not found for sample_id={args.sample_id}: {args.h5ad}")

    resolved_mask_name = format_mask_name(args.mask_name, args.mask_seed, args.model_seed)
    run_name = args.run_name or default_run_name(
        args.sample_id, args.mask_seed, args.model_seed, args.seed, args.epochs, resolved_mask_name
    )
    run_root = os.path.join(PROJECT_ROOT, "results", run_name)
    os.makedirs(run_root, exist_ok=True)

    mask_dir = ensure_masks(args, run_root)
    train_mask_path = os.path.join(mask_dir, "train_mask.npy")
    val_mask_path = os.path.join(mask_dir, "val_mask.npy")

    spatial_dir = os.path.join(run_root, "baselines")
    spatial_model_paths = OrderedDict([
        ("mean", ("Mean Baseline", os.path.join(spatial_dir, "mean_baseline_ratio.npy"))),
        ("softimpute", ("SoftImpute", os.path.join(spatial_dir, "softimpute_ratio.npy"))),
        ("multiple_imputation", ("Multiple Imputation", os.path.join(spatial_dir, "mi_ratio.npy"))),
        ("spatial_knn", ("Spatial KNN", os.path.join(spatial_dir, "spatial_knn_ratio.npy"))),
        ("spatial_idw", ("Spatial IDW", os.path.join(spatial_dir, "spatial_idw_ratio.npy"))),
    ])
    need_spatial = any(key in selected_models for key in spatial_model_paths)
    if need_spatial and not (args.skip_existing and all(
        prediction_file_valid(path, mask_dir)
        for key, (_label, path) in spatial_model_paths.items()
        if key in selected_models
    )):
        run_cmd([
            PYTHON,
            os.path.join(PROJECT_ROOT, "src", "models", "baselines", "train.py"),
            "--h5ad", args.h5ad,
            "--mask_dir", mask_dir,
            "--out_dir", spatial_dir,
            "--threshold", "10",
            "--min_depth", "10",
            "--n_jobs", str(cpu_threads),
        ], cwd=PROJECT_ROOT, dry_run=args.dry_run, label="Spatial/Matrix baselines")

    neural_jobs = [
        ("AIRGate-ST", "AIRGate-ST", os.path.join(PROJECT_ROOT, "src", "models", "AIRGate-ST", "train.py"), []),
        ("AIRDiff-ST", "AIRDiff-ST", os.path.join(PROJECT_ROOT, "src", "models", "AIRDiff-ST", "train.py"), []),
    ]
    for label, key, script, extra_args in neural_jobs:
        if key not in selected_models:
            continue
        out_base = os.path.join(run_root, key)
        pred_path = os.path.join(out_base, f"seed{args.model_seed}", "pred_ratio.npy")
        if args.skip_existing and prediction_file_valid(pred_path, mask_dir):
            print(f"[Skip] {key}: {pred_path}")
            continue
        run_cmd([
            PYTHON,
            script,
            *extra_args,
            "--epochs", str(args.epochs),
            "--seed", str(args.model_seed),
            "--device", args.device,
            "--data_path", args.h5ad,
            "--output_dir", out_base,
            "--train_mask_path", train_mask_path,
            "--val_mask_path", val_mask_path,
        ], cwd=PROJECT_ROOT, dry_run=args.dry_run, label=label)

    registry_path = os.path.join(run_root, "model_registry.csv")
    rows = [
        item
        for key, item in spatial_model_paths.items()
        if key in selected_models
    ]
    rows.extend(
        (label, os.path.join(run_root, key, f"seed{args.model_seed}", "pred_ratio.npy"))
        for label, key, _script, _extra_args in neural_jobs
        if key in selected_models
    )
    if not args.dry_run:
        write_registry(registry_path, rows)
    else:
        print(f"[DryRun] would write registry -> {registry_path}")

    metrics_dir = os.path.join(run_root, "metrics")
    run_cmd([
            PYTHON,
            os.path.join(PROJECT_ROOT, "evaluation", "collect_layered_metrics.py"),
        "--mask_dir", mask_dir,
        "--out_dir", metrics_dir,
        "--model_registry", registry_path,
    ], cwd=PROJECT_ROOT, dry_run=args.dry_run, label="Layered metrics")

    if not args.skip_dual_block:
        dual_block_dir = os.path.join(run_root, "dual_block_heatmaps")
        run_cmd([
            PYTHON,
            os.path.join(PROJECT_ROOT, "visualization", "generate_all_dual_blocks.py"),
            "--model_registry", registry_path,
            "--mask_dir", mask_dir,
            "--out_dir", dual_block_dir,
        ], cwd=PROJECT_ROOT, dry_run=args.dry_run, label="Dual-block heatmaps")

    print(f"[Done] run_root={run_root}")


if __name__ == "__main__":
    main()
