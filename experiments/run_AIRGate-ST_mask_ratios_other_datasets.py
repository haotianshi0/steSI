import os
import subprocess
import sys
from datetime import datetime


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

JOBS = [
    ("151507", 2874),
    ("151508", 2874),
    ("151509", 2874),
    ("151510", 2874),
    ("151669", 1531),
    ("151670", 1531),
    ("151671", 1531),
    ("151672", 1531),
    ("151674", 1531),
    ("151675", 1531),
    ("151676", 1531),
]
RATIOS = [20, 40]


def stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main() -> int:
    epochs = 100
    device = "cuda"
    os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)

    for sample_id, seed in JOBS:
        for ratio in RATIOS:
            mask_name = f"per_site_random_{ratio}_seed{{seed}}"
            log_prefix = f"AIRGate-ST_ratio_{sample_id}_{ratio}_seed{seed}"
            out_log = os.path.join(PROJECT_ROOT, "results", f"{log_prefix}.out.log")
            err_log = os.path.join(PROJECT_ROOT, "results", f"{log_prefix}.err.log")

            cmd = [
                sys.executable,
                os.path.join(PROJECT_ROOT, "experiments", "run_harder_benchmark.py"),
                "--sample_id", sample_id,
                "--seed", str(seed),
                "--mask_seed", str(seed),
                "--model_seed", str(seed),
                "--epochs", str(epochs),
                "--device", device,
                "--mask_name", mask_name,
                "--models", "AIRGate-ST",
                "--skip_dual_block",
                "--skip_existing",
                "--cpu_threads", "0",
            ]

            with open(out_log, "a", encoding="utf-8") as out, open(err_log, "a", encoding="utf-8") as err:
                out.write(f"\n[{stamp()}] START sample={sample_id} mask={ratio}% seed={seed}\n")
                out.write("[Command] " + " ".join(cmd) + "\n")
                out.flush()
                rc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=out, stderr=err).returncode
                if rc != 0:
                    out.write(f"[{stamp()}] FAILED sample={sample_id} mask={ratio}% exit_code={rc}\n")
                    out.flush()
                    return rc
                out.write(f"[{stamp()}] DONE sample={sample_id} mask={ratio}% seed={seed}\n")
                out.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
