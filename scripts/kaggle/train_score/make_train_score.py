"""Write one Kaggle kernel directory per train-and-score arm from train_score.py.

Usage: ``python make_train_score.py [arm ...]`` writes the named arms, or every arm when none
is named. Naming arms leaves the kernels of earlier arms as they were generated. An arm in
SESSIONS gets two kernels, ``ts_<arm>_p1`` and ``ts_<arm>_p2``; push p2 only once p1 has a
completed version, since p2 reads p1's output.
"""

import json
import sys
from pathlib import Path

COMMIT = "d05ea19"
#: E1 (FINDINGS §171): fp16, split attention, DDP on a T4x2 at 4 tasks per rank (global batch 8)
E1 = ["--amp", "fp16", "--attn-impl", "split", "--loader", "stream", "--ddp", "--batch-size", "4"]
#: E1's 48k arm at seed 0: 6,000 steps x 8 tasks
S0_48K = ["--seed", "0", "--steps", "6000"]
#: DR2.4 (FINDINGS §180). The DR2.3 knobs at scripts/prior_stats.py's V2_KNOBS, the values
#: their checks passed at; p_binary 0.7 is left out because it is inert at max_classes 2.
BANDS = "0.5:0:0.05,0.35:0.05:0.2,0.15:0.2:0.5"
KNOBS = ["--feature-ratio-bands", BANDS, "--imbalance-draws", "--categorical-mode", "unordered",
         "--missing-mode", "mixed", "--tie-column-prob", "0.15", "--irrelevant-frac-max", "0.2",
         "--duplicate-frac-max", "0.1", "--label-flip-max", "0.05"]
#: Financial v2: half the labels gated, each ratio's sign a per-task coin (DR2.2), and the
#: width bands, the only knob the financial source reads. The published recipe's sequential
#: p_financial 1.0 draws every task from it.
FIN_V2 = ["--financial-gate-prob", "0.5", "--financial-sign-flip-prob", "0.5",
          "--feature-ratio-bands", BANDS]
#: The general trunk: prior_stats.py's "v2" mixture without its financial share. --p-financial
#: goes back to PriorConfig's default 0.7, since a weighted draw refuses a non-default p_*.
TRUNK_WEIGHTS = "scm=0.3,tree=0.2,task_family=0.1,gated=0.1"
TRUNK = ["--p-financial", "0.7", "--source-weights", TRUNK_WEIGHTS, *KNOBS]
INTERLEAVED = ["--p-financial", "0.7", "--source-weights", f"{TRUNK_WEIGHTS},financial=0.3",
               "--financial-gate-prob", "0.5", "--financial-sign-flip-prob", "0.5", *KNOBS]
#: DR7.4 (FINDINGS §185): the multiclass control v4-multiclass.pt's prior (§99: --max-classes
#: 10 --p-financial 0.5, so half the tasks are the generic SCM's, uniform on 2 to 10 classes)
#: with the SCM sampler it was trained on (--scm-legacy: it predates 48.17/48.19's widening of
#: 2026-09-21), plus AU5.5's per-task class permutation, the one prior change under test
MULTICLASS = ["--max-classes", "10", "--p-financial", "0.5", "--scm-legacy", "--permute-classes"]
ARMS = {
    "e1-v48-s0": (E1 + ["--seed", "0", "--steps", "6000"], True),
    "e1-v48-s1": (E1 + ["--seed", "1", "--steps", "6000"], True),
    "e1-v480-s0": (E1 + ["--seed", "0", "--steps", "60000"], True),
    "e1-v480-s1": (E1 + ["--seed", "1", "--steps", "60000"], True),
    # DR6.2 / DR6.3 screening (FINDINGS §175): E1's 48k recipe at seed 0, one lever each,
    # against E1's 48k arms as controls
    "dr62-induced-s0": (E1 + ["--seed", "0", "--steps", "6000", "--cell-row-attn", "induced"],
                        True),
    "dr63-cls-s0": (E1 + ["--seed", "0", "--steps", "6000", "--readout", "cls", "--head-mlp"],
                    True),
    # DR2.4 prior v2 A/B (FINDINGS §180): E1's 48k recipe at seed 0, against e1-v48-s0
    "dr24-a-s0": (E1 + S0_48K + FIN_V2, True),  # financial v2 alone
    "dr24-b-s0": (E1 + S0_48K + TRUNK, True),  # general trunk alone
    "dr24-c-s0": (E1 + S0_48K + TRUNK, True),  # trunk, then credit from SWITCH below
    "dr24-d-s0": (E1 + S0_48K + INTERLEAVED, True),  # trunk and financial v2 in one draw
    # DR7.4 (FINDINGS §185): E1's 48k recipe at seed 0 with the multiclass prior, against
    # v4-multiclass.pt on the proxy suite's multiclass tasks
    "dr74-mc-s0": (E1 + S0_48K + MULTICLASS, True),
    # DR7.4 seed-1 replicate, §185's pre-declared next step for a candidate (§186)
    "dr74-mc-s1": (E1 + ["--seed", "1", "--steps", "6000"] + MULTICLASS, True),
    # DR7.4 volume step (FINDINGS §198): the candidate recipe at E1's 480,000 tasks
    "dr74-mc-v480-s0": (E1 + ["--seed", "0", "--steps", "60000"] + MULTICLASS, True),
    # DR8.2 model size at volume (FINDINGS §184): e1-v480-s0's recipe with twice the layers,
    # d_model, heads and d_ff unchanged, against e1-v480-s0
    "dr82-depth8-s0": (E1 + ["--seed", "0", "--steps", "60000", "--n-layers", "8"], True),
    # DR8.2 width (FINDINGS §199): d_model 192 at 6 heads in the row stages, cell stage kept at
    # 4 heads, 1.88M parameters against the depth arm's 1.68M
    "dr82-width192-s0": (E1 + ["--seed", "0", "--steps", "60000", "--d-model", "192", "--d-ff", "768",
                               "--n-heads", "6", "--n-cell-heads", "4"], True),
}
#: Arms trained but not scored on V4FinBench, which is binary: a multiclass arm is decided
#: on the proxy suite, run off Kaggle
TRAIN_ONLY = {"dr74-mc-s0", "dr74-mc-s1", "dr74-mc-v480-s0"}
#: Two-session arms (DR8.2): session 1's training wall-time budget in seconds. Each writes two
#: kernels, <arm>-p1 (trains for the budget with --run-seconds, no scoring) and <arm>-p2
#: (attaches p1's output as a kernel source, resumes to --steps and scores). 30,600 s is
#: 8.5 h, which leaves a 12 h session room for the install and the final save.
SESSIONS = {"dr82-depth8-s0": 30600, "dr82-width192-s0": 30600}
#: Curriculum arms: the step the prior switches at, and the second phase's flags. 4,200 of
#: 6,000 steps on the trunk gives arm c arm d's 0.3 financial share, so c and d differ in
#: order only.
SWITCH = {"dr24-c-s0": (4200, E1 + S0_48K + FIN_V2)}
#: Arms pinned at another commit: 6916a89 adds the DR6.2/DR6.3 flags on top of d05ea19 and
#: nothing else under src/fintfm/modeling or src/fintfm/prior, with defaults unchanged; the
#: DR2.4 commit adds --new-prior-phase, and at the recipe's prior config draws the same
#: training stream and trains the same weights as d05ea19 (FINDINGS §180)
DR24_COMMIT = "5e0fad0"
#: DR8.2's commit adds --run-seconds, whose default reads no clock and changes no weight
DR82_COMMIT = "e493123"
COMMITS = {"dr62-induced-s0": "6916a89", "dr63-cls-s0": "6916a89",
           **{arm: DR24_COMMIT for arm in ARMS if arm.startswith("dr24-")},
           # DR7.4 needs no code change: a901cdc (origin/main, 2026-10-24) carries every flag
           "dr74-mc-s0": "a901cdc",
           "dr74-mc-s1": "a901cdc",
           "dr74-mc-v480-s0": "a901cdc",
           "dr82-depth8-s0": DR82_COMMIT,
           "dr82-width192-s0": DR82_COMMIT}
OLD_ARMS = {
    # DR4.4 replicates of the published recipe (seed 0 also re-checks reproducibility)
    "pub-s0": ["--seed", "0"],
    "pub-s1": ["--seed", "1"],
    "pub-s2": ["--seed", "2"],
    # AU2.8: the published recipe without the cell-attention block, same prior and protocol
    "nocell-s0": ["--seed", "0", "--n-cell-blocks", "0", "--no-cell-labels"],
}
HERE = Path(__file__).parent
TEMPLATE = (HERE / "train_score.py").read_text()

wanted = sys.argv[1:] or list(ARMS)
unknown = sorted(set(wanted) - set(ARMS))
if unknown:
    sys.exit(f"unknown arm(s) {unknown}; known: {list(ARMS)}")
clash = sorted(set(SESSIONS) & set(SWITCH))
if clash:
    sys.exit(f"arm(s) {clash} are both two-session and curriculum; the template runs one or other")
for arm in wanted:
    flags, ddp = ARMS[arm]
    switch, phase2 = SWITCH.get(arm, (0, []))
    # (kernel name suffix, session, session 1's kernel id or None): one session, or two
    sessions: list[tuple[str, int, str | None]] = [("", 0, None)]
    if arm in SESSIONS:
        first = f"muhakabartay/fintfm-ts-{arm}-p1"
        sessions = [("-p1", 1, None), ("-p2", 2, first)]
    for suffix, session, source in sessions:
        name = f"{arm}{suffix}"
        d = HERE.parent / f"ts_{name.replace('-', '_')}"
        d.mkdir(exist_ok=True)
        code = (TEMPLATE.replace("__COMMIT__", COMMITS.get(arm, COMMIT))
                .replace("__ARM__", arm)
                .replace("[]  # __FLAGS__", json.dumps(flags))
                .replace("False  # __DDP__", str(ddp)).replace("False  # __PART__", "False")
                .replace("0  # __SWITCH__", str(switch))
                .replace("[]  # __PHASE2__", json.dumps(phase2))
                .replace("0  # __SESSION__", str(session))
                .replace("0  # __RUN_SECONDS__", str(SESSIONS.get(arm, 0)))
                .replace('""  # __SESSION1__', json.dumps(source or ""))
                .replace("True  # __SCORE__", str(arm not in TRAIN_ONLY)))
        (d / "kernel.py").write_text(code)
        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": f"muhakabartay/fintfm-ts-{name}", "title": f"fintfm-ts-{name}",
            "code_file": "kernel.py", "language": "python", "kernel_type": "script",
            "is_private": "true", "enable_gpu": "true", "enable_tpu": "false",
            "enable_internet": "true",
            "dataset_sources": ["sebastiantomczak10/v4-group-corporate-bankruptcy"],
            "competition_sources": [], "kernel_sources": [source] if source else [],
            "model_sources": [],
        }, indent=2) + "\n")
        print(d)
