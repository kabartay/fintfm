"""Kaggle kernel template: pretrain the published recipe with one change, then score it.

Design review DR4.4 (noise yardstick from matched replicates) and audit AU2.8 (cell attention
at a matched prior). Each arm trains `configs/published/v4-cellattn-labels.yaml` (48,000 tasks,
fp32, masked attention, single GPU -- the published checkpoint's own protocol) at the
commit pinned below, with ARM_FLAGS appended, then scores the checkpoint zero-shot on
V4FinBench's five full-panel folds with the packaged default inference settings, as the
§160 matched control was scored. Outputs: the checkpoint, its training log, and per-fold
predictions with v4_protocol.json, in /kaggle/working/<ARM>/.

A curriculum arm (design review DR2.4, arm c) sets SWITCH_STEP: it trains ARM_FLAGS for
SWITCH_STEP steps of the same --steps schedule (`--run-steps`), keeps that checkpoint as
`<ARM>.phase1.pt`, then resumes the state with PHASE2_FLAGS and `--new-prior-phase` for the
rest. Optimiser, schedule and both RNG streams carry over, so only the prior changes.

A train-only arm (design review DR7.4, a multiclass checkpoint) sets SCORE_V4 False.
V4FinBench is binary, so such an arm is decided elsewhere (the proxy suite's multiclass tasks,
run off Kaggle), and the kernel stops after training without reading the V4FinBench data.

A run longer than one Kaggle session (design review DR8.2) is two kernels, SESSION 1 and 2.
Session 1 trains ARM_FLAGS with `--run-seconds RUN_SECONDS`, which stops at a step boundary
and writes `<ARM>.pt` and `<ARM>.pt.state`, and skips scoring. Session 2 attaches session 1's
kernel output (kernel-metadata `kernel_sources`, mounted read-only under /kaggle/input),
copies the two files into /kaggle/working/<ARM>/ as `<ARM>.session1.pt` and
`<ARM>.session1.pt.state`, resumes the state to the full `--steps` with the same ARM_FLAGS,
then scores as usual. Session 2 finds the state by globbing /kaggle/input for
`<ARM>.pt.state` and stops unless there is exactly one, since the mount layout of a kernel
source is not pinned down here. SESSION 0 is the one-session run every other arm is.

`make_train_score.py` writes one copy of this file per arm with ARM and ARM_FLAGS filled in.
"""

import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

COMMIT = "e493123"
ARM = "dr82-depth8-s0"
ARM_FLAGS: list[str] = ["--amp", "fp16", "--attn-impl", "split", "--loader", "stream", "--ddp", "--batch-size", "4", "--seed", "0", "--steps", "60000", "--n-layers", "8"]
DDP = True
SCORE_PARTITION = False
SWITCH_STEP = 0
PHASE2_FLAGS: list[str] = []
SCORE_V4 = True
SESSION = 2
RUN_SECONDS = 30600
SESSION1_KERNEL = "muhakabartay/fintfm-ts-dr82-depth8-s0-p1"
RECIPE_URL = (f"https://raw.githubusercontent.com/kabartay/fintfm/{COMMIT}/"
              "configs/published/v4-cellattn-labels.yaml")


def find_session1_state(arm: str, inputs: Path = Path("/kaggle/input")) -> Path:
    """The training state session 1 wrote, from session 1's attached kernel output.

    Args:
        arm: The arm name; session 1 wrote ``<arm>.pt.state`` beside ``<arm>.pt``.
        inputs: The read-only input mount to search.

    Returns:
        The one ``<arm>.pt.state`` found anywhere under ``inputs``, whose ``<arm>.pt`` sibling
        exists.

    Raises:
        SystemExit: Unless exactly one state file is found and its checkpoint sits beside it,
            naming every candidate, so a wrong attachment fails before any GPU time is spent.
    """
    hits = sorted(inputs.rglob(f"{arm}.pt.state"))
    if len(hits) != 1:
        listing = sorted(str(p) for p in inputs.rglob("*") if p.is_file())[:200]
        sys.exit(f"expected one {arm}.pt.state under {inputs} (session 1's output, kernel "
                 f"source {SESSION1_KERNEL!r}), found {hits}; files there: {listing}")
    ckpt = hits[0].with_suffix("")  # <arm>.pt.state -> <arm>.pt
    if not ckpt.is_file():
        sys.exit(f"{hits[0]} has no {ckpt.name} beside it; session 1 did not finish its save")
    return hits[0]


def main() -> None:
    """Train the arm (or this session's share of it), then score it on V4FinBench."""
    if SESSION not in (0, 1, 2):
        sys.exit(f"SESSION must be 0, 1 or 2, not {SESSION}")
    if SESSION and SWITCH_STEP:
        sys.exit("a two-session arm cannot also switch priors; the generator refuses both")
    if SESSION == 1 and RUN_SECONDS <= 0:
        sys.exit("session 1 needs a RUN_SECONDS budget")
    if SESSION and not SCORE_V4:
        sys.exit("a two-session arm is scored on V4FinBench in session 2; it cannot be train-only")
    root = None
    if SCORE_V4:
        hits = sorted(Path("/kaggle/input").rglob("company_years_h1.parquet"))
        if len(hits) != 1:
            sys.exit(f"expected one company_years_h1.parquet under /kaggle/input, found {hits}")
        root = hits[0].parent
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True, check=False)
    if smi.returncode != 0 or not smi.stdout.strip():
        sys.exit("no GPU in this session (quota exhausted?) -- refusing a CPU run")
    print(f"GPUs: {smi.stdout.strip()}; data: {root}", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                    f"fintfm @ git+https://github.com/kabartay/fintfm.git@{COMMIT}"], check=True)
    out = Path("/kaggle/working") / ARM
    out.mkdir(parents=True, exist_ok=True)
    recipe = out / "recipe.yaml"
    urllib.request.urlretrieve(RECIPE_URL, recipe)
    ckpt = out / f"{ARM}.pt"
    launcher = [sys.executable, "-u", "-m", "fintfm.modeling.train"]
    if DDP:
        launcher = [sys.executable, "-u", "-m", "torch.distributed.run", "--standalone",
                    "--nproc-per-node", "2", "-m", "fintfm.modeling.train"]
    base = [*launcher, "--config", str(recipe), "--device", "cuda",
            "--checkpoint-every", "1000", "--out", str(ckpt)]
    phases = [[*base, *ARM_FLAGS]]
    if SESSION == 1:
        phases = [[*base, *ARM_FLAGS, "--run-seconds", str(RUN_SECONDS)]]
    elif SESSION == 2:
        found = find_session1_state(ARM)
        state = out / f"{ARM}.session1.pt.state"
        shutil.copy(found, state)
        shutil.copy(found.with_suffix(""), out / f"{ARM}.session1.pt")
        log1 = found.parent / "train.log"
        if log1.is_file():
            shutil.copy(log1, out / "train.session1.log")
        print(f"session 1's state: {found} -> {state}", flush=True)
        phases = [[*base, *ARM_FLAGS, "--resume", str(state)]]
    if SWITCH_STEP:
        phases = [[*base, *ARM_FLAGS, "--run-steps", str(SWITCH_STEP)],
                  [*base, *PHASE2_FLAGS, "--resume", f"{ckpt}.state", "--new-prior-phase"]]
    for i, train in enumerate(phases):
        print("$ " + " ".join(train), flush=True)
        with (out / "train.log").open("a" if i else "w") as log:
            rc = subprocess.run(train, stdout=log, stderr=subprocess.STDOUT,
                                check=False).returncode
        print((out / "train.log").read_text()[-3000:], flush=True)
        if rc != 0:
            sys.exit(f"training phase {i + 1} failed with exit {rc}")
        if i + 1 < len(phases):
            shutil.copy(ckpt, out / f"{ARM}.phase1.pt")  # the trunk-only weights at the switch
    from fintfm.provenance import read_checkpoint_provenance

    rec = read_checkpoint_provenance(ckpt) or {}
    print(f"checkpoint: {rec.get('steps_completed')} steps, {rec.get('tasks_seen')} tasks; "
          f"time budget: {rec.get('time_budget')}", flush=True)
    if SESSION == 1:
        for p in out.glob(f"{ARM}.pt.step*"):
            p.unlink()  # session 2 needs only <ARM>.pt and <ARM>.pt.state
        if not (out / f"{ARM}.pt.state").is_file():
            sys.exit(f"session 1 wrote no {ARM}.pt.state; session 2 would have nothing to resume")
        print("session 1 done, not scored; session 2 attaches this kernel's output", flush=True)
        return
    if not SCORE_V4:
        _keep_final(out)
        print("done (train only: V4FinBench scoring skipped)")
        return
    from fintfm.config import load_config
    from fintfm.experiments.v4_protocol import run, summarise

    # the cell-attention arms ran out of memory scoring at the default chunks on a 14.56 GB T4
    # (2026-10-18); these chunks are identity-preserving, so no score changes
    score_cfg = out / "score.yaml"
    score_cfg.write_text("inference:\n  query_chunk: 512\n  feature_chunk: 8\n")
    cfg = load_config(score_cfg)
    record = run(str(ckpt), out / "score", horizon=0, folds=(0, 1, 2, 3, 4), root=root,
                 with_boosting=False, classical=("logistic_regression",), cfg=cfg, device="cuda")
    print(summarise(record), flush=True)
    if SCORE_PARTITION:
        part_cfg = out / "score_partition.yaml"
        part_cfg.write_text(score_cfg.read_text() + "large_n:\n  method: partition\n")
        record = run(str(ckpt), out / "score_partition", horizon=0, folds=(0, 1, 2, 3, 4),
                     root=root, with_boosting=False, classical=("logistic_regression",),
                     cfg=load_config(part_cfg), device="cuda")
        print(summarise(record), flush=True)
    _keep_final(out)
    print("done")


def _keep_final(out: Path) -> None:
    """Delete the intermediate step checkpoints, keeping the final one.

    Args:
        out: The arm's output directory.
    """
    for p in out.glob(f"{ARM}.pt.step*"):
        p.unlink()


if __name__ == "__main__":
    main()
