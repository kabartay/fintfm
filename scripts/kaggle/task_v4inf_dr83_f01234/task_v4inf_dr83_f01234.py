"""Kaggle kernel: V4FinBench sanity and cost of the 512-column binary checkpoint (DR8.3 Steps 0 and 2).

The launch gate for the multiclass 512-column arm (docs/results/drafts/dr8.3_2026-11.md, section
7, confirmed by the owner on 2026-11-07) needs two numbers from the binary arm `dr83-width512-s0`:

- **Step 0, cost.** One V4FinBench fold on a T4 takes at most 4 times `e1-v48-s0`'s fold. That
  fold took about 426 s (MEASURED from `runs/kaggle-ts-e1-v48-s0/fintfm-ts-e1-v48-s0.log`, the
  same scoring call below), so the limit is 1,704 s. Fold 0 runs alone first and the kernel stops
  there if it is over, so a failing Step 0 spends one fold of quota, not five.
- **Step 2, sanity.** Five-fold mean AP at least 0.1810, no Holm-significant fold loss; read
  locally with `scripts/compare_v4_scores.py --cluster` against `e1-v48-s0`.

The scoring call is the train-and-score template's (scripts/kaggle/ts_dr83_width512_s0/kernel.py):
the packaged default inference settings with the identity-preserving memory chunks
`query_chunk` 512 and `feature_chunk` 8, logistic regression as the classical reference, no
boosting. The checkpoint comes from the training kernel's output (`kernel_sources`), verified by
SHA-256 against the local copy fetched on 2026-11-14. Pinned to 62a73c3, the training commit.
PyTorch's fused encoder fast path is switched off (see ``main``) so fold 0 fits a T4.
Outputs: `/kaggle/working/dr83_v4/f0/` (fold 0) and `/kaggle/working/dr83_v4/f1234/`, each with
per-fold predictions and `v4_protocol.json`; `timing.json` holds the fold-0 seconds.
"""

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

COMMIT = "62a73c3"
ARM = "dr83-width512-s0"
CKPT_SHA256 = "20381bf13adaf122f3c8c5d461b755c3ab718f4a73c959a6b7e80105cae19a27"
CONTROL_FOLD_S = 426.0  # e1-v48-s0, fold 0 on a T4 (MEASURED, see docstring)
STEP0_FACTOR = 4.0
#: Folds still to score after fold 0 ran (2026-11-14). The first run's guard timed the whole
#: call, data loading (about 300 s) included, while the control's 426 s starts after loading:
#: 1,877 s against 1,704 s read as a fail, but the fold itself took 1,581 s (327 s to 1,908 s
#: in the kernel log), 3.7 times the control, inside the 4x limit. So Step 0 passed and only
#: folds 1-4 run here; set to () to repeat the fold-0 timing.
REMAINING_FOLDS: tuple[int, ...] = (1, 2, 3, 4)
CONFIG = "inference:\n  query_chunk: 512\n  feature_chunk: 8\n"


def find_checkpoint(inputs: Path = Path("/kaggle/input")) -> Path:
    """The training kernel's checkpoint, from its attached output.

    Args:
        inputs: The read-only input mount to search.

    Returns:
        The one ``<ARM>.pt`` found under ``inputs``.

    Raises:
        SystemExit: Unless exactly one is found, naming the candidates.
    """
    hits = sorted(inputs.rglob(f"{ARM}.pt"))
    if len(hits) != 1:
        sys.exit(f"expected one {ARM}.pt under {inputs}, found {hits}")
    return hits[0]


def score(root: Path, ckpt: Path, out_dir: Path, folds: tuple[int, ...]) -> float:
    """Score ``folds`` with the template's call and return the wall-clock seconds.

    Args:
        root: Directory holding ``company_years_h1.parquet``.
        ckpt: The checkpoint file.
        out_dir: Where predictions and ``v4_protocol.json`` are written.
        folds: Folds to score.

    Returns:
        Seconds the call took, data loading included (as in the control's log).
    """
    from fintfm.config import load_config
    from fintfm.experiments.v4_protocol import run, summarise

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir.with_suffix(".yaml")
    cfg_path.write_text(CONFIG)
    start = time.time()
    record = run(str(ckpt), out_dir, horizon=0, folds=folds, root=root, with_boosting=False,
                 classical=("logistic_regression",), cfg=load_config(cfg_path), device="cuda")
    seconds = time.time() - start
    print(summarise(record), flush=True)
    print(f"folds {folds}: {seconds:.0f} s", flush=True)
    return seconds


def main() -> None:
    """Find data and checkpoint, install the pinned fintfm, time fold 0, then score the rest."""
    hits = sorted(Path("/kaggle/input").rglob("company_years_h1.parquet"))
    if len(hits) != 1:
        sys.exit(f"expected one company_years_h1.parquet under /kaggle/input, found {hits}")
    root = hits[0].parent
    ckpt = find_checkpoint()
    digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    if digest != CKPT_SHA256:
        sys.exit(f"checkpoint hash {digest} != {CKPT_SHA256}")
    print(f"data: {root}; checkpoint: {ckpt}", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                    f"fintfm @ git+https://github.com/kabartay/fintfm.git@{COMMIT}"], check=True)
    import torch

    if not torch.cuda.is_available():
        sys.exit("no GPU in this session (quota exhausted?) -- refusing a CPU run")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    # The first launch (2026-11-14) ran out of memory on fold 0: in eval mode nn.TransformerEncoder
    # takes its fused fast path, which builds the full 512 x 512 column-attention matrix (9.81 GiB
    # asked of a 14.56 GiB T4). Off, attention goes through scaled_dot_product_attention's
    # memory-efficient kernel. Same predictions: max abs difference 2.4e-7 on a toy table with
    # this checkpoint on CPU (SMOKE-TEST), float32 rounding.
    torch.backends.mha.set_fastpath_enabled(False)
    out = Path("/kaggle/working/dr83_v4")
    if REMAINING_FOLDS:
        score(root, ckpt, out / "f1234", REMAINING_FOLDS)
        print("done")
        return
    f0 = score(root, ckpt, out / "f0", (0,))
    limit = STEP0_FACTOR * CONTROL_FOLD_S
    passed = f0 <= limit
    (out / "timing.json").write_text(json.dumps(
        {"fold0_seconds": f0, "control_fold_seconds": CONTROL_FOLD_S, "limit_seconds": limit,
         "step0_passed": passed}, indent=1))
    if not passed:
        print(f"Step 0 FAILED: fold 0 took {f0:.0f} s > {limit:.0f} s; folds 1-4 not scored")
        return
    print(f"Step 0 passed: {f0:.0f} s <= {limit:.0f} s", flush=True)
    score(root, ckpt, out / "f1234", (1, 2, 3, 4))
    print("done")


if __name__ == "__main__":
    main()
