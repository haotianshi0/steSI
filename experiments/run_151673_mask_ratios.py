"""Sweep the main reporting mask ratios (20%, 40%, 60%) on sample 151673.

Driver script that calls ``run_harder_benchmark.py`` once per ratio with a
fixed seed (1531) and the full 7-model set, redirecting stdout / stderr to
per-ratio log files under ``results/``. Used to produce the per-mask-ratio
``layered_metrics.csv`` files consumed by
``visualization/plot_selected_models_mask_ratio.py``.
"""

import os
import subprocess
import sys
from datetime import datetime


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def stamp() -> str:
    """Return the current local time as ``YYYY-MM-DD HH:MM:SS`` for log lines."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main() -> int:
    """Run the 20% / 40% / 60% mask-ratio sweep for sample 151673.

    Aborts on the first non-zero return code from any benchmark invocation.
    Returns the subprocess exit code (0 on full success).
    """
    ratios = [20, 40, 60]
    seed = 1531
    epochs = 100
    device = "cuda"

    for ratio in ratios:
        mask_name = f"per_site_random_{ratio}_seed{{seed}}"
        out_log = os.path.join(PROJECT_ROOT, "results", f"mask_ratio_151673_{ratio}_seed{seed}.out.log")
        err_log = os.path.join(PROJECT_ROOT, "results", f"mask_ratio_151673_{ratio}_seed{seed}.err.log")
        os.makedirs(os.path.dirname(out_log), exist_ok=True)

        cmd = [
            sys.executable,
            os.path.join(PROJECT_ROOT, "experiments", "run_harder_benchmark.py"),
            "--sample_id", "151673",
            "--seed", str(seed),
            "--mask_seed", str(seed),
            "--model_seed", str(seed),
            "--epochs", str(epochs),
            "--device", device,
            "--mask_name", mask_name,
            "--models", "all",
            "--skip_dual_block",
            "--skip_existing",
            "--cpu_threads", "0",
        ]

        with open(out_log, "a", encoding="utf-8") as out, open(err_log, "a", encoding="utf-8") as err:
            out.write(f"\n[{stamp()}] START 151673 mask={ratio}% seed={seed} epochs={epochs} device={device}\n")
            out.write("[Command] " + " ".join(cmd) + "\n")
            out.flush()
            rc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=out, stderr=err).returncode
            if rc != 0:
                out.write(f"[{stamp()}] FAILED 151673 mask={ratio}% exit_code={rc}\n")
                out.flush()
                return rc
            out.write(f"[{stamp()}] DONE 151673 mask={ratio}% seed={seed}\n")
            out.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
