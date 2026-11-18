"""Diagnose which device FinTFM-LinProbe uses on an HF Jobs GPU, and how long a T4 fit takes.

The reproduction kernel fitted Amazon_employee_access in 1,821 s on an HF t4-medium, the same as
on CPU, against 610 s in the stored Kaggle T4 run, which forced ``FINTFM_DEVICE=cuda`` (2026-11-18).
This job installs exactly what the kernel installs (it imports the generated kernel and reuses its
``clone_fork`` and ``install``), then fits two tables with the registry entry's default config,
printing what ``FinTFMModel._fit`` receives (``num_gpus``, ``num_cpus``) and the device every
checkpoint load uses:

1. ``blood-transfusion-service-center`` with no ``FINTFM_*`` variable (fast; the device question);
2. ``Amazon_employee_access`` with ``FINTFM_DEVICE=cuda`` (the timing question: near 610 s means
   the device explains the slowdown).

Launch like ``repro_kernel_on_hf_job.py`` (``KERNEL`` and ``FINTFM_REF`` in the environment), on
``t4-small`` with a short timeout. Prints only; uploads nothing. ``DIAG_DATASETS`` (comma-separated)
replaces the two fits with default-config fits of those tables, and ``AG_OVERRIDE`` reinstalls the
four AutoGluon packages at that version after the kernel's install, for an A/B on the nightly.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import types
import urllib.request
from pathlib import Path

WORKER = """\
import sys, time
import tabarena.models.fintfm.model as fm
import fintfm.modeling.model as mm

_fit = fm.FinTFMModel._fit
def fit(self, *a, **k):
    print(f"DIAG _fit num_gpus={k.get('num_gpus')} num_cpus={k.get('num_cpus')}", flush=True)
    return _fit(self, *a, **k)
fm.FinTFMModel._fit = fit

_load = mm.FinancialTFM.load.__func__
@classmethod
def load(cls, *a, **k):
    print(f"DIAG checkpoint load map_location={k.get('map_location')}", flush=True)
    return _load(cls, *a, **k)
mm.FinancialTFM.load = load

from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.contexts import TabArenaContext

dataset = sys.argv[1]
t0 = time.time()
TabArenaContext().build_and_run_jobs(
    TabArenaV0pt1ExperimentBundle(models=[("FinTFM-LinProbe", 0)]).build_experiments(),
    expname="/tmp/diag_results",
    subset="lite",
    build_kwargs={"dataset_names": [dataset]},
    new_result_prefix="[New] ",
    debug_mode=True,
)
print(f"DIAG {dataset}: {time.time() - t0:.0f} s", flush=True)
"""


def load_kernel(kernel: str, ref: str) -> types.ModuleType:
    """Import the generated kernel script at ``ref`` as a module, without running its main.

    Args:
        kernel: Kernel directory name under ``scripts/kaggle/``.
        ref: fintfm git commit.

    Returns:
        The kernel module.
    """
    url = f"https://raw.githubusercontent.com/kabartay/fintfm/{ref}/scripts/kaggle/{kernel}/{kernel}.py"
    mod = types.ModuleType("kernel")
    mod.__dict__["__name__"] = "kernel"
    # The generated kernel at a pinned commit of this repository, run as a module: trusted source.
    code = compile(urllib.request.urlopen(url, timeout=60).read().decode(), url, "exec")
    exec(code, mod.__dict__)  # noqa: S102
    return mod


def main() -> int:
    """Clone and install as the kernel does, then run the two diagnostic fits."""
    k = load_kernel(os.environ["KERNEL"], os.environ["FINTFM_REF"])
    fork = Path("/tmp/tabarena")
    k.clone_fork(fork)
    k.install(fork)
    ag = os.environ.get("AG_OVERRIDE")
    if ag:  # the A/B on the AutoGluon nightly: replace the four packages, dependencies untouched
        pkgs = [f"autogluon.{p}=={ag}" for p in ("tabular", "core", "common", "features")]
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--no-deps", *pkgs], check=True
        )
        print(f"DIAG AutoGluon overridden to {ag}", flush=True)
    worker = Path("/tmp/diag_worker.py")
    worker.write_text(WORKER)
    base = {n: v for n, v in os.environ.items() if not n.startswith("FINTFM_")}
    runs = (
        [(d, {}) for d in os.environ["DIAG_DATASETS"].split(",")]
        if os.environ.get("DIAG_DATASETS")
        else [
            ("blood-transfusion-service-center", {}),
            ("Amazon_employee_access", {"FINTFM_DEVICE": "cuda"}),
        ]
    )
    for dataset, extra in runs:
        print(f"DIAG start {dataset} {extra}", flush=True)
        t0 = time.time()
        code = subprocess.run(
            [sys.executable, "-P", str(worker), dataset],
            cwd=fork,
            env={**base, **extra},
            check=False,
        ).returncode
        print(f"DIAG done {dataset}: exit {code}, {time.time() - t0:.0f} s wall", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
