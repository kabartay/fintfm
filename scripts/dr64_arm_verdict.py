"""DR6.4: check a fetched pretraining-arm output, write its scoring command, judge it, compose.

Three steps, each its own subcommand, so the verdict is read off rather than argued:

* ``prepare`` -- for one fetched kernel output directory (``runs/kaggle-<kernel-id>/``): find its
  one checkpoint, check the kernel log for a GPU (``CUDA available: True``, or the ``GPUs:`` line
  the ``ts_*`` kernels print) and the arm's expected lines, and write
  ``dr64_score_command.sh``, the CPU V4FinBench scoring command with the control's settings.
  Most ``ts_*`` kernels already score on the Kaggle GPU (``<ARM>/score/``); the command is for
  those that do not (DR8.3 width 512 is train-only, the AU5.9 arms run a probe) and for a re-score.
* ``verdict`` -- after scoring, compare the arm's five folds with the same-seed control by
  ``compare_v4_scores.compare`` (paired company-level bootstrap) and apply §175's rule: mean AP
  difference above :data:`YARDSTICK` with at least :data:`MIN_FOLDS` of 5 folds positive is
  promising, the mirror is harmful, otherwise null. Writes ``dr64_verdict.json`` in the directory.
* ``compose`` -- DR6.4's composition rule over every verdict: a lever enters only if promising at
  seed 0 against ``e1-v48-s0`` and at seed 1 against ``e1-v48-s1`` and harmful at neither.

The thresholds are constants on purpose (as in ``c5_verdict.py``): moving them after a result is
what pre-registration forbids. The arm is read from the kernel, not from a table kept here:
``scripts/kaggle/ts_*/kernel.py`` carries ``ARM`` and ``ARM_FLAGS``.

Usage:
    uv run --frozen python scripts/dr64_arm_verdict.py prepare \\
        <repo>/runs/kaggle-fintfm-ts-au517-lenscale-s0
    uv run --frozen python scripts/dr64_arm_verdict.py verdict \\
        <repo>/runs/kaggle-fintfm-ts-au517-lenscale-s0
    uv run --frozen python scripts/dr64_arm_verdict.py compose \\
        <repo>/runs
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shlex
import sys
from pathlib import Path

import numpy as np
from compare_v4_scores import compare

#: §173, fixed by §175: the pretraining noise yardstick, five-fold mean AP.
YARDSTICK = 0.0117
#: §175: folds, out of five, that must share the mean's sign.
MIN_FOLDS = 4
#: Execution flags every ts_* arm shares with the control; they are not the lever.
EXEC_FLAGS = {"--amp", "--attn-impl", "--loader", "--ddp", "--batch-size", "--seed", "--steps"}
#: Valued execution flags (the others above are followed by one value, ``--ddp`` by none).
EXEC_NOVALUE = {"--ddp"}
#: Arms that are controls of another arm, not levers: left out of the composition.
CONTROL_ARMS = ("au59-hazard-ctrl",)
REPO = Path(__file__).resolve().parents[1]
KERNELS = REPO / "scripts" / "kaggle"
DEFAULT_RUNS = Path(__file__).resolve().parents[1] / "runs"
#: Extra train.log lines a flag must produce, beyond the shared ones (train.py's `say` calls).
FLAG_LOG_LINES = {"--masked-column-prob": "masked-column swaps"}


def lever_of(arm: str) -> str:
    """The lever an arm belongs to: its name without the seed suffix.

    Args:
        arm: e.g. ``au517-lenscale-s1``.

    Returns:
        e.g. ``au517-lenscale``.
    """
    return re.sub(r"-s\d+$", "", arm)


def read_kernel(arm: str, kernels: Path = KERNELS) -> dict:
    """Read ``ARM_FLAGS`` and ``SCORE_V4`` of the kernel that trains ``arm``.

    Args:
        arm: The arm name a kernel's ``ARM`` constant holds.
        kernels: Directory holding the ``ts_*`` kernel directories.

    Returns:
        ``{"dir", "flags", "score_v4"}``; ``score_v4`` is True when the kernel scores itself.

    Raises:
        SystemExit: If no kernel, or more than one, trains this arm.
    """
    hits = []
    for f in sorted(kernels.glob("ts_*/kernel.py")):
        consts: dict = {}
        for node in ast.parse(f.read_text()).body:
            tgt = None
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                tgt = node.targets[0]
            elif isinstance(node, ast.AnnAssign):
                tgt = node.target
            if isinstance(tgt, ast.Name) and tgt.id in ("ARM", "ARM_FLAGS", "SCORE_V4"):
                try:
                    consts[tgt.id] = ast.literal_eval(node.value)
                except ValueError:
                    pass
        if consts.get("ARM") == arm:
            hits.append({"dir": f.parent, "flags": list(consts.get("ARM_FLAGS", [])),
                         "score_v4": bool(consts.get("SCORE_V4", False))})
    if len(hits) != 1:
        raise SystemExit(f"expected one kernel under {kernels} with ARM = {arm!r}, found {len(hits)}")
    return hits[0]


def flag_value(flags: list[str], name: str) -> str | None:
    """The value following ``name`` in a flag list, or ``None`` if absent."""
    return flags[flags.index(name) + 1] if name in flags[:-1] else None


def lever_flags(flags: list[str]) -> list[str]:
    """The flags of an arm beyond the execution flags every ts_* arm shares.

    Args:
        flags: An arm's ``ARM_FLAGS``.

    Returns:
        The lever's tokens, in order (a flag with its value, or a bare switch).
    """
    out: list[str] = []
    i = 0
    while i < len(flags):
        tok = flags[i]
        step = 1 if tok in EXEC_NOVALUE or (i + 1 < len(flags) and flags[i + 1].startswith("--")) else 2
        if tok not in EXEC_FLAGS:
            out.extend(flags[i:i + step])
        i += step
    return out


def find_checkpoint(out_dir: Path) -> Path:
    """The one final checkpoint under a fetched output directory.

    Args:
        out_dir: ``runs/kaggle-<kernel-id>/``.

    Returns:
        The ``.pt`` file, ignoring ``.step<N>``, ``.phase1`` and ``.session1`` intermediates.

    Raises:
        SystemExit: Unless exactly one remains.
    """
    skip = (".phase1.pt", ".session1.pt")
    hits = [p for p in sorted(out_dir.rglob("*.pt")) if not p.name.endswith(skip)]
    if len(hits) != 1:
        raise SystemExit(f"{out_dir}: expected one checkpoint, found {[str(h) for h in hits]}")
    return hits[0]


def check_logs(out_dir: Path, ckpt: Path, kernel: dict) -> list[str]:
    """Check the kernel log and train.log for a GPU run and the arm's expected lines.

    The ``ts_*`` kernels refuse a CPU run and print ``GPUs: <name>``; kernels from
    ``make_real_arm_kernel.py`` print ``CUDA available: True``. Either shows a GPU; an explicit
    ``CUDA available: False`` fails. The arm's expected lines come from its own flags: its command
    echo in the kernel log carries every flag, and train.log carries the DDP line, the last step,
    and any line a flag adds (``FLAG_LOG_LINES``). Flags with no distinct line (optimiser, value
    embedding, cell blocks) are checked by the echo only.

    Args:
        out_dir: The fetched directory.
        ckpt: Its checkpoint; the arm's directory holds ``train.log``.
        kernel: ``read_kernel``'s result.

    Returns:
        Failures, empty when every check passed.
    """
    fails: list[str] = []
    logs = sorted(out_dir.glob("*.log"))
    text = "\n".join(p.read_text(errors="replace") for p in logs)
    if not logs:
        fails.append("no kernel log (*.log) in the fetched directory")
    elif "CUDA available: False" in text:
        fails.append("kernel log says CUDA available: False")
    elif "CUDA available: True" not in text and not re.search(r"GPUs: [^\\\n]*\S", text):
        fails.append("kernel log shows neither `CUDA available: True` nor a `GPUs:` line")
    flags = kernel["flags"]
    echo = " ".join(flags)
    if echo not in text.replace('"\n,{"stream_name":"stdout","time"', " "):
        fails.append(f"kernel log lacks the command echo with the arm's flags: {echo}")
    train_log = ckpt.parent / "train.log"
    if not train_log.is_file():
        fails.append(f"no train.log beside {ckpt.name}")
        return fails
    t = train_log.read_text(errors="replace")
    steps = flag_value(flags, "--steps")
    batch = flag_value(flags, "--batch-size")
    if steps and f"step {steps}/{steps}" not in t:
        fails.append(f"train.log never reaches `step {steps}/{steps}`")
    if batch and "--ddp" in flags and f"ddp: 2 processes, {int(batch) * 2} tasks per step" not in t:
        fails.append(f"train.log lacks `ddp: 2 processes, {int(batch) * 2} tasks per step`")
    if "model parameters:" not in t:
        fails.append("train.log lacks `model parameters:`")
    for flag, line in FLAG_LOG_LINES.items():
        if flag in flags and line not in t:
            fails.append(f"train.log lacks `{line}` (expected from {flag})")
    return fails


def control_dir(runs: Path, seed: int) -> Path:
    """The scored output of ``e1-v48-s<seed>``.

    Args:
        runs: The runs root.
        seed: 0 or 1.

    Returns:
        The control's ``score`` directory (predictions and ``v4_protocol.json``).
    """
    return runs / f"kaggle-ts-e1-v48-s{seed}" / f"e1-v48-s{seed}" / "score"


def score_command(ctrl: Path, ckpt: Path, out: Path) -> str:
    """The V4FinBench scoring command with the control's settings, read from its config.

    Reads ``v4_protocol.json``'s ``config`` for horizon, folds, classical arms, boosting and
    device, and the control's ``score.yaml`` (inference chunks, identity-preserving) beside it.
    A control setting the entry point cannot take as a flag (fine-tuning, ensembling, a large-n
    wrapper) makes this refuse, since the command would then score under other settings.

    Args:
        ctrl: The control's ``score`` directory.
        ckpt: The checkpoint to score.
        out: Where ``fintfm-v4protocol`` writes.

    Returns:
        A one-line shell command, run from the repo root.

    Raises:
        SystemExit: If the control's record is missing or uses settings the flags cannot express.
    """
    rec_file = ctrl / "v4_protocol.json"
    if not rec_file.is_file():
        raise SystemExit(f"{rec_file} missing: cannot read the control's settings")
    cfg = json.loads(rec_file.read_text())["config"]
    odd = {k: cfg.get(k) for k, want in (("finetune_steps", 0), ("n_ensemble", 1),
                                         ("max_rows", None), ("tuned", False))
           if cfg.get(k) != want}
    if (cfg.get("large_n") or {}).get("method", "none") != "none":
        odd["large_n"] = cfg["large_n"]["method"]
    if odd:
        raise SystemExit(f"control settings not expressible as flags: {odd}")
    parts = ["nice", "-n", "19", "uv", "run", "--frozen", "fintfm-v4protocol", "--model", str(ckpt),
             "--horizon", str(cfg["horizon"]), "--folds", ",".join(map(str, cfg["folds"]))]
    if not cfg.get("boosting"):
        parts.append("--no-boosting")
    parts += ["--classical", ",".join(cfg.get("classical") or ["logistic_regression"]),
              "--device", "cpu"]
    yaml = ctrl.parent / "score.yaml"
    if yaml.is_file():
        parts += ["--config", str(yaml)]
    parts += ["--out", str(out)]
    return "PYTHONUNBUFFERED=1 " + shlex.join(parts)


def judge(diffs: list[float]) -> str:
    """§175's rule on five per-fold AP differences (arm minus control).

    Args:
        diffs: Per-fold differences.

    Returns:
        ``"promising"`` if the mean exceeds :data:`YARDSTICK` with at least :data:`MIN_FOLDS`
        folds positive; ``"harmful"`` under the mirror; otherwise ``"null"``. Boundaries are
        strict, as in `c5_verdict.py`.
    """
    mean = float(np.mean(diffs))
    if mean > YARDSTICK and sum(d > 0 for d in diffs) >= MIN_FOLDS:
        return "promising"
    if mean < -YARDSTICK and sum(d < 0 for d in diffs) >= MIN_FOLDS:
        return "harmful"
    return "null"


def find_scores(out_dir: Path, ckpt: Path) -> Path | None:
    """The arm's scored directory: the Kaggle GPU score, else the local CPU score.

    Args:
        out_dir: The fetched directory.
        ckpt: Its checkpoint.

    Returns:
        The directory holding ``predictions_fold*.npz``, or ``None`` if neither exists.
    """
    for d in (ckpt.parent / "score", out_dir / "score_cpu", out_dir / "score.json"):
        if d.is_dir() and list(d.glob("predictions_fold*.npz")):
            return d
    return None


#: Where an arm's capability-probe record may sit under its fetched directory, in order.
PROBE_FILES = ("probe/capability.json", "capability.json", "capability/capability.json")


def teaching_axis(out_dir: Path, arm: str) -> dict:
    """The arm's antisymmetric-probe AUC, the "teaching" axis that sits beside protocol AP.

    ``docs/results/FINDINGS.md`` §63 showed a prior can transfer well (high V4FinBench AP) while
    teaching badly (low antisymmetric-probe AUC), so a verdict that carried AP alone would hide
    it. The probe record is the ``capability.json`` written by ``fintfm.experiments.capability``.

    Args:
        out_dir: The fetched directory.
        arm: The arm (checkpoint stem), as named in the probe record.

    Returns:
        ``{"antisymmetric_auc": float | None, "symmetric_auc": {probe: float}, "source": str}``.
        The AUC is ``None`` and ``source`` says why when no record or no row for the arm exists:
        a missing axis is stated, never silently dropped.
    """
    for base, rel in ((b, r) for b in (out_dir / arm, out_dir) for r in PROBE_FILES):
        f = base / rel
        if not f.is_file():
            continue
        rows = [r for r in json.loads(f.read_text())["results"] if r["arm"] == arm]
        by = {r["probe"]: float(r["auc_mean"]) for r in rows}
        if "antisymmetric" in by:
            return {"antisymmetric_auc": by["antisymmetric"],
                    "symmetric_auc": {k: by[k] for k in ("symmetric_sum", "symmetric_count")
                                      if k in by},
                    "source": str(f)}
        return {"antisymmetric_auc": None, "symmetric_auc": {},
                "source": f"{f}: no antisymmetric row for arm {arm}"}
    return {"antisymmetric_auc": None, "symmetric_auc": {},
            "source": "not run: no capability.json under " + str(out_dir)}


def cmd_prepare(args: argparse.Namespace) -> int:
    """Run the ``prepare`` subcommand; returns the exit status."""
    out_dir: Path = args.out_dir
    ckpt = find_checkpoint(out_dir)
    arm = ckpt.stem
    kernel = read_kernel(arm)
    seed = int(flag_value(kernel["flags"], "--seed") or 0)
    fails = check_logs(out_dir, ckpt, kernel)
    cmd = score_command(control_dir(args.runs, seed), ckpt, out_dir / "score_cpu")
    (out_dir / "dr64_score_command.sh").write_text("#!/usr/bin/env bash\ncd " + shlex.quote(str(REPO))
                                                  + f"\n{cmd}\n")
    scored = find_scores(out_dir, ckpt)
    print(f"arm {arm} (seed {seed}), checkpoint {ckpt}")
    print(f"kernel scores itself on Kaggle: {kernel['score_v4']}; scored predictions found: "
          f"{scored if scored else 'none (run dr64_score_command.sh, one scoring at a time)'}")
    print(f"command written to {out_dir / 'dr64_score_command.sh'}")
    for f in fails:
        print(f"FAIL: {f}")
    print("checks: " + ("FAILED" if fails else "passed"))
    return 1 if fails else 0


def cmd_verdict(args: argparse.Namespace) -> int:
    """Run the ``verdict`` subcommand; returns the exit status."""
    out_dir: Path = args.out_dir
    ckpt = find_checkpoint(out_dir)
    arm = ckpt.stem
    kernel = read_kernel(arm)
    seed = int(flag_value(kernel["flags"], "--seed") or 0)
    scored = args.score_dir or find_scores(out_dir, ckpt)
    if scored is None:
        raise SystemExit(f"{out_dir}: no scored predictions yet; run dr64_score_command.sh")
    ctrl = control_dir(args.runs, seed)
    try:
        rows = compare(ctrl, scored, n_boot=args.n_boot, cluster=True)
        boot = "company"
    except SystemExit as e:
        if "company" not in str(e):
            raise
        rows = compare(ctrl, scored, n_boot=args.n_boot, cluster=False)
        boot = "row"
    diffs = [r["diff"] for r in rows]
    if len(diffs) != 5:
        raise SystemExit(f"{scored}: {len(diffs)} folds scored, the rule needs five")
    ckernel = read_kernel(f"e1-v48-s{seed}")
    differs = sorted({t for t in lever_flags(kernel["flags"]) if t.startswith("--")}
                     | {f for f in ("--batch-size", "--steps")
                        if flag_value(kernel["flags"], f) != flag_value(ckernel["flags"], f)})
    rec = {
        "arm": arm, "lever": lever_of(arm), "seed": seed, "control": f"e1-v48-s{seed}",
        "scored_dir": str(scored), "bootstrap": boot, "yardstick": YARDSTICK,
        "min_folds": MIN_FOLDS, "verdict": judge(diffs), "mean_diff": float(np.mean(diffs)),
        "folds_positive": int(sum(d > 0 for d in diffs)),
        "folds_negative": int(sum(d < 0 for d in diffs)),
        "mean_ap_arm": float(np.mean([r["arm"] for r in rows])),
        "mean_ap_control": float(np.mean([r["control"] for r in rows])),
        "folds": [{k: (list(v) if isinstance(v, tuple) else v) for k, v in r.items()}
                  for r in rows],
        "teaching_axis": teaching_axis(out_dir, arm),
        "lever_flags": lever_flags(kernel["flags"]),
        "differs_from_control": differs,
        "recipe_matches_control_execution": all(
            flag_value(kernel["flags"], f) == flag_value(ckernel["flags"], f)
            for f in ("--batch-size", "--steps")),
    }
    (out_dir / "dr64_verdict.json").write_text(json.dumps(rec, indent=2, default=float) + "\n")
    print(f"{arm}: mean AP {rec['mean_ap_arm']:.4f} vs {rec['mean_ap_control']:.4f}, "
          f"diff {rec['mean_diff']:+.4f}, {rec['folds_positive']}/5 positive "
          f"({boot}-level bootstrap) -> {rec['verdict'].upper()}")
    ta = rec["teaching_axis"]
    print(f"{arm}: antisymmetric-probe AUC "
          + (f"{ta['antisymmetric_auc']:.4f}" if ta["antisymmetric_auc"] is not None
             else f"MISSING ({ta['source']})") + " beside the protocol AP above (§63)")
    if not rec["recipe_matches_control_execution"]:
        print("note: batch size or steps differ from the control's; a confound the verdict "
              "does not remove (dr8.3_2026-11.md)")
    return 0


def compose(verdicts: list[dict], expect: list[str] | None = None,
            not_composable: tuple[str, ...] = ()) -> dict:
    """DR6.4's composition rule over per-arm verdicts.

    A lever enters if promising at seed 0 against ``e1-v48-s0`` and at seed 1 against
    ``e1-v48-s1`` and harmful at neither, and is composable. Two or more entering levers give the
    bundle, one a single-lever test, none no bundle. A lever promising at seed 0 with no seed-1
    verdict, or an expected lever with no seed-0 verdict, is pending, and the outcome is then
    ``"wait"``: the bundle is fixed only when every arm has reported.

    Args:
        verdicts: ``dr64_verdict.json`` contents.
        expect: Levers that must have reported; absent ones are pending.
        not_composable: Levers whose flags cannot be set together on one commit.

    Returns:
        ``{"outcome", "levers": {lever: {"status", "reason"}}, "bundle": [levers]}`` with outcome
        ``"bundle"``, ``"single-lever"``, ``"none"`` or ``"wait"``.
    """
    by: dict[str, dict[int, str]] = {}
    for v in verdicts:
        if v["lever"] in CONTROL_ARMS:
            continue
        by.setdefault(v["lever"], {})[v["seed"]] = v["verdict"]
    for lever in expect or []:
        by.setdefault(lever, {})
    levers: dict[str, dict] = {}
    for lever, seeds in sorted(by.items()):
        s0, s1 = seeds.get(0), seeds.get(1)
        if "harmful" in seeds.values():
            bad = next(s for s, v in seeds.items() if v == "harmful")
            st, why = "excluded", f"harmful at seed {bad}"
        elif s0 is None:
            st, why = "pending", "no seed-0 verdict"
        elif s0 != "promising":
            st, why = "excluded", f"seed 0 {s0}"
        elif s1 is None:
            st, why = "pending", "promising at seed 0; seed-1 replicate not reported"
        elif s1 != "promising":
            st, why = "excluded", f"seed 1 {s1}"
        elif lever in not_composable:
            st, why = "excluded", "not composable"
        else:
            st, why = "qualifies", "promising at both seeds, harmful at neither"
        levers[lever] = {"status": st, "reason": why}
    bundle = [lv for lv, r in levers.items() if r["status"] == "qualifies"]
    if any(r["status"] == "pending" for r in levers.values()):
        outcome = "wait"
    else:
        outcome = "bundle" if len(bundle) >= 2 else "single-lever" if bundle else "none"
    return {"outcome": outcome, "levers": levers, "bundle": bundle}


def cmd_compose(args: argparse.Namespace) -> int:
    """Run the ``compose`` subcommand; returns the exit status."""
    files = sorted(args.runs.glob("kaggle-*/dr64_verdict.json"))
    verdicts = [json.loads(f.read_text()) for f in files]
    expect = args.expect.split(",") if args.expect else None
    res = compose(verdicts, expect, tuple(filter(None, args.not_composable.split(","))))
    res["verdict_files"] = [str(f) for f in files]
    (args.runs / "dr64_composition.json").write_text(json.dumps(res, indent=2) + "\n") \
        if args.write else None
    print(f"{len(verdicts)} verdict file(s)")
    for lever, r in res["levers"].items():
        print(f"  {lever}: {r['status']} ({r['reason']})")
    print(f"outcome: {res['outcome']}" + (f"; bundle = {res['bundle']}" if res["bundle"] else ""))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    for name in ("prepare", "verdict"):
        s = sub.add_parser(name)
        s.add_argument("out_dir", type=Path)
        s.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
        if name == "verdict":
            s.add_argument("--score-dir", type=Path, default=None)
            s.add_argument("--n-boot", type=int, default=2000)
    c = sub.add_parser("compose")
    c.add_argument("runs", type=Path)
    c.add_argument("--expect", default="", help="comma-separated levers that must have reported")
    c.add_argument("--not-composable", default="", help="comma-separated levers to exclude")
    c.add_argument("--write", action="store_true", help="write dr64_composition.json in runs")
    args = p.parse_args(argv)
    return {"prepare": cmd_prepare, "verdict": cmd_verdict, "compose": cmd_compose}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
