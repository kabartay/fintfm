"""SMOKE-TEST of the V4FinBench inference kernels' ``run_arm`` on a tiny synthetic panel, on CPU.

Proves each kernel's embedded configuration loads under the installed fintfm and that
``fintfm.experiments.v4_protocol.run`` completes one fold with it. Says nothing about quality:
the panel is random noise with a planted weak signal and the real data never enters. Run with
``PYTHONPATH=<src of the pinned commit>`` to test against the commit a kernel installs.

Usage::

    nice -n 19 uv run --frozen --extra bench python scripts/kaggle/smoke_v4_inference_kernels.py \
        --ckpt runs/v4-cellattn-labels.pt --out /path/to/scratch
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def synthetic_panel(path: Path, n_companies: int = 400, n_features: int = 12) -> None:
    """Write a small panel in V4FinBench's column layout (company, country, main_label, numerics).

    Args:
        path: Output parquet path.
        n_companies: Companies, each with 1-5 rows.
        n_features: Numeric feature columns.
    """
    rng = np.random.RandomState(0)
    rows = []
    for k in range(n_companies):
        for _ in range(rng.randint(1, 6)):
            x = rng.randn(n_features)
            p = 1.0 / (1.0 + np.exp(-(x[0] * 1.5 - 4.0)))
            rows.append({"company": f"c{k}", "country": f"C{k % 4}",
                         "main_label": int(rng.rand() < p),
                         **{f"f{j}": float(v) for j, v in enumerate(x)}})
    pd.DataFrame(rows).to_parquet(path)


def main() -> None:
    """Run every kernel's ``run_arm`` on fold 0 of a synthetic panel and report the seconds."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--only", default="", help="run only kernels whose name contains this")
    args = ap.parse_args()
    import fintfm
    print("fintfm from", Path(fintfm.__file__).parent, flush=True)
    work = args.out or Path(tempfile.mkdtemp())
    panel = work / "panel"
    panel.mkdir(parents=True, exist_ok=True)
    synthetic_panel(panel / "company_years_h1.parquet")
    failed = []
    for d in sorted(HERE.glob(f"task_v4inf_*{args.only}*")):
        spec = importlib.util.spec_from_file_location(d.name, d / f"{d.name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        t = time.time()
        try:
            mod.run_arm(panel, args.ckpt, work / d.name, (0,), "cpu")
            print(f"SMOKE-TEST {d.name}: ok, {time.time() - t:.1f} s", flush=True)
        except Exception as e:  # noqa: BLE001 - report every kernel, not only the first failure
            failed.append(d.name)
            print(f"SMOKE-TEST {d.name}: FAILED {type(e).__name__}: {e}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
