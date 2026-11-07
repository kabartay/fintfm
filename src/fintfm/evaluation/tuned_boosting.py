"""Parent-side launcher for the tuned LightGBM worker (DR8.4).

:mod:`fintfm.evaluation.lightgbm_tuned` must run in a process that never imports torch, so
callers that do (the credit suite, the V4FinBench hybrid) hand it arrays through an npz and
read back its JSON. This module is that hand-off, shared so the credit driver and the V4FinBench
hybrid launch the same search the same way.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

#: The worker script; run by path so ``sys.path[0]`` is its directory and ``fintfm`` is not
#: imported (and with it torch).
WORKER = Path(__file__).resolve().parent / "lightgbm_tuned.py"


def run_tuned(
    arrays: dict[str, np.ndarray], n_configs: int = 30, budget_s: float | None = 720.0,
    seed: int = 0, threads: int = 2, timeout: float | None = None, predictions: bool = False,
) -> dict[str, Any] | None:
    """Tune and score one split in the torch-free worker.

    Args:
        arrays: The worker's npz contents: ``Xtr, ytr, Xte, yte``, optionally ``ttr``, ``gtr``,
            ``embargo`` and the starting scores ``btr, bte`` (see
            :func:`fintfm.evaluation.lightgbm_tuned.main`).
        n_configs: Configs to try (config 0 is the untuned control's).
        budget_s: Search seconds after which no new config starts; ``None`` for no limit.
        seed: Seeds the search, the inner split and every fit.
        threads: OpenMP threads for the worker.
        timeout: Seconds before the worker is abandoned.
        predictions: Also return the tuned model's probabilities for every ``Xte`` row, under
            the key ``"predictions"`` (an array, so not for ``json.dump``).

    Returns:
        The worker's JSON (tuned and untuned test AP and ROC-AUC, the chosen config and every
        trial), or ``None`` if the worker failed, which is printed.
    """
    with tempfile.TemporaryDirectory() as tmp:
        npz, out = Path(tmp) / "d.npz", Path(tmp) / "r.json"
        pred = Path(tmp) / "p.npy"
        np.savez(npz, **arrays)
        argv = ["nice", "-n", "19", sys.executable, str(WORKER), str(npz), str(out),
                str(n_configs), "inf" if budget_s is None else str(budget_s), str(seed)]
        if predictions:
            argv.append(str(pred))
        proc = subprocess.run(
            argv, capture_output=True, text=True, check=False, timeout=timeout,
            env={**os.environ, "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)},
        )
        if proc.returncode != 0 or not out.exists():
            tail = (proc.stderr or "").strip().splitlines()
            print(f"  tuned lightgbm failed: {tail[-1][:160] if tail else proc.returncode}",
                  flush=True)
            return None
        res = json.loads(out.read_text())
        if predictions:
            res["predictions"] = np.load(pred)
        return res
