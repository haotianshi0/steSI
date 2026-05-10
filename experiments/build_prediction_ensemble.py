"""Build prediction-level ensembles by averaging across model seeds.

Companion to ``run_seed_ensemble.py``: given the same fixed mask seed and
several model seeds, this script loads each method's per-seed
``pred_ratio.npy`` files, averages the finite predictions per cell,
preserves training-visible ground-truth values, and writes one ensemble
prediction file per method into ``--out_dir/predictions/``. A new
``model_registry.csv`` is emitted so the ensemble predictions can be
scored by ``evaluation/collect_layered_metrics.py`` exactly like a
single-seed run.

Invoked automatically by ``run_seed_ensemble.py``; can also be run
standalone if the per-seed run directories already exist.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from typing import Dict, List

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON = sys.executable


def parse_seeds(value: str) -> List[int]:
    """Parse a comma-separated seed list into a list of ints (rejects empty)."""
    seeds = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not seeds:
        raise ValueError("--model_seeds must contain at least one integer")
    return seeds


def format_mask_name(mask_name: str, mask_seed: int, model_seed: int) -> str:
    """Substitute seed placeholders in a mask-name template."""
    return mask_name.format(seed=mask_seed, mask_seed=mask_seed, model_seed=model_seed)


def run_name_for(sample_id: str, mask_seed: int, model_seed: int, epochs: int, mask_name: str) -> str:
    """Compose the per-(mask_seed, model_seed) run directory name."""
    resolved_mask = format_mask_name(mask_name, mask_seed, model_seed)
    sample_prefix = "" if sample_id == "151673" else f"{sample_id}_"
    return f"{sample_prefix}mask{mask_seed}_model{model_seed}_e{epochs}_{resolved_mask}"


def read_registry(path: str) -> List[Dict[str, str]]:
    """Load a ``method,pred_path`` registry CSV into a list of row dicts."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_registry(path: str, rows: List[tuple[str, str]]) -> None:
    """Write a ``method,pred_path`` CSV listing the ensembled predictions."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "pred_path"])
        writer.writerows(rows)
    print(f"[Registry] {path}")


def safe_name(name: str) -> str:
    """Sanitise a method label into a filesystem-safe lowercase identifier."""
    out = []
    for ch in name.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in (" ", "-", "_", "/", "(", ")"):
            out.append("_")
    compact = "".join(out).strip("_")
    while "__" in compact:
        compact = compact.replace("__", "_")
    return compact or "model"


def same_mask_or_raise(reference_dir: str, candidate_dir: str) -> None:
    """Confirm two mask directories agree on train/val/observed/gt arrays.

    Used to guarantee that all per-seed runs can be safely averaged: if the
    masks differ at all, ensembling would mix predictions over different
    holdout cells and silently bias the metrics.
    """
    for filename in ["train_mask.npy", "val_mask.npy", "observed_mask.npy", "gt_ratio.npy"]:
        a = np.load(os.path.join(reference_dir, filename))
        b = np.load(os.path.join(candidate_dir, filename))
        if not np.array_equal(a, b, equal_nan=True):
            raise ValueError(f"Mask mismatch for {filename}: {reference_dir} vs {candidate_dir}")


def finite_mean_stack(arrays: List[np.ndarray]) -> np.ndarray:
    """Per-cell average over an iterable of arrays, ignoring NaNs.

    Cells where every input is NaN remain NaN in the output. Used to combine
    per-seed predictions into a single ensemble matrix.
    """
    stack = np.stack(arrays, axis=0).astype(np.float32)
    finite = np.isfinite(stack)
    count = finite.sum(axis=0)
    total = np.where(finite, stack, 0.0).sum(axis=0)
    out = np.full(stack.shape[1:], np.nan, dtype=np.float32)
    np.divide(total, count, out=out, where=count > 0)
    return out


def run_cmd(cmd: List[str], cwd: str, label: str) -> None:
    """Echo and run a subprocess command; abort on non-zero exit."""
    print(f"[Run] {label}: " + " ".join(f'"{x}"' if " " in x else x for x in cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)
    print(f"[Done] {label}", flush=True)


def main() -> None:
    """CLI entry point: ensemble per-seed predictions and (optionally) score them.

    Locates the per-(mask_seed, model_seed) run directories, validates that
    every seed used the same mask, averages each method's prediction across
    seeds, writes the ensembled ``pred_ratio.npy`` files plus a new
    ``model_registry.csv``, and finally invokes
    ``evaluation/collect_layered_metrics.py`` on the ensemble registry
    unless ``--skip_metrics`` is set.
    """
    parser = argparse.ArgumentParser(description="Build prediction-level ensembles across model seeds for a fixed mask seed.")
    parser.add_argument("--mask_seed", type=int, required=True)
    parser.add_argument("--model_seeds", required=True)
    parser.add_argument("--sample_id", default="151673", help="Dataset/sample id used in run directories and mask directories.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--mask_name", default="per_site_random_60_seed{seed}")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--skip_metrics", action="store_true")
    args = parser.parse_args()

    model_seeds = parse_seeds(args.model_seeds)
    run_items = []
    for model_seed in model_seeds:
        run_name = run_name_for(args.sample_id, args.mask_seed, model_seed, args.epochs, args.mask_name)
        run_root = os.path.join(PROJECT_ROOT, "results", run_name)
        mask_dir = os.path.join(
            run_root,
            f"masks_{args.sample_id}_seed{args.mask_seed}",
            format_mask_name(args.mask_name, args.mask_seed, model_seed),
        )
        registry_path = os.path.join(run_root, "model_registry.csv")
        if not os.path.exists(registry_path):
            raise FileNotFoundError(f"Missing model registry: {registry_path}")
        run_items.append({
            "model_seed": model_seed,
            "run_name": run_name,
            "run_root": run_root,
            "mask_dir": mask_dir,
            "registry": read_registry(registry_path),
        })

    reference_mask_dir = run_items[0]["mask_dir"]
    for item in run_items[1:]:
        same_mask_or_raise(reference_mask_dir, item["mask_dir"])

    gt = np.load(os.path.join(reference_mask_dir, "gt_ratio.npy")).astype(np.float32)
    train_mask = np.load(os.path.join(reference_mask_dir, "train_mask.npy")).astype(bool)

    os.makedirs(args.out_dir, exist_ok=True)
    pred_dir = os.path.join(args.out_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    first_methods = [row["method"] for row in run_items[0]["registry"]]
    registry_by_seed = {
        item["model_seed"]: {row["method"]: row["pred_path"] for row in item["registry"]}
        for item in run_items
    }

    ensemble_rows = []
    summary_rows = []
    for method in first_methods:
        missing = [seed for seed, reg in registry_by_seed.items() if method not in reg]
        if missing:
            raise ValueError(f"{method}: missing from registries for model seeds {missing}")

        preds = []
        source_paths = []
        for item in run_items:
            path = registry_by_seed[item["model_seed"]][method]
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing prediction for {method}, model_seed={item['model_seed']}: {path}")
            pred = np.load(path).astype(np.float32)
            if pred.shape != gt.shape:
                raise ValueError(f"{method}: shape mismatch {pred.shape} != {gt.shape}")
            preds.append(pred)
            source_paths.append(os.path.abspath(path))

        ensemble = finite_mean_stack(preds)
        ensemble[train_mask] = gt[train_mask]
        finite = np.isfinite(ensemble)
        if np.any(finite & ((ensemble < 0.0) | (ensemble > 1.0))):
            raise ValueError(f"{method}: ensemble finite predictions outside [0,1]")

        out_path = os.path.join(pred_dir, f"{safe_name(method)}_ensemble_pred_ratio.npy")
        np.save(out_path, ensemble)
        ensemble_rows.append((f"{method} Ensemble", out_path))
        summary_rows.append({
            "method": method,
            "ensemble_method": f"{method} Ensemble",
            "pred_path": os.path.abspath(out_path),
            "source_count": len(preds),
            "source_paths": source_paths,
            "finite_count": int(finite.sum()),
            "nan_count": int((~finite).sum()),
        })
        print(f"[Ensemble] {method} -> {out_path}")

    registry_path = os.path.join(args.out_dir, "model_registry.csv")
    write_registry(registry_path, ensemble_rows)

    summary_path = os.path.join(args.out_dir, "prediction_ensemble_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "mask_seed": args.mask_seed,
            "sample_id": args.sample_id,
            "model_seeds": model_seeds,
            "mask_dir": os.path.abspath(reference_mask_dir),
            "registry": os.path.abspath(registry_path),
            "models": summary_rows,
        }, f, indent=2)
    print(f"[Saved] {summary_path}")

    if not args.skip_metrics:
        run_cmd([
            PYTHON,
            os.path.join(PROJECT_ROOT, "evaluation", "collect_layered_metrics.py"),
            "--mask_dir", reference_mask_dir,
            "--out_dir", os.path.join(args.out_dir, "metrics"),
            "--model_registry", registry_path,
        ], PROJECT_ROOT, "ensemble layered metrics")


if __name__ == "__main__":
    main()
