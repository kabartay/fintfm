# Experiment configurations

Override files for the command-line entry points. Each is **deep-merged over the packaged
default** at [`src/fintfm/configs/default.yaml`](../src/fintfm/configs/default.yaml), so a
file here only carries what it changes — copying the whole default would freeze every other
value at the moment of copying, and then quietly diverge.

```bash
uv run fintfm-ctxsweep --model runs/v4-hazard-ldp.pt --config configs/context-sweep-3seed.yaml
uv run fintfm-v4oot    --model runs/v4-hazard-ldp.pt --config configs/v4-oot-quick.yaml
FINTFM_CONFIG=configs/context-sweep-3seed.yaml uv run fintfm-ctxsweep --model runs/m.pt
```

Resolution order, lowest priority first: **packaged default → `--config` (or `$FINTFM_CONFIG`)
→ explicit CLI flag.** Every run prints the layers it used and records them in its output
JSON as `config_sources`, so a number can be traced back to the settings that produced it.

## What belongs in configuration, and what does not

Not every literal is configuration, and treating them all as such makes a system less robust
rather than more. The rule used here:

> A value belongs in configuration if a reasonable experiment would want it different. A value
> whose change would make results incomparable, or the code incorrect, stays in code.

**In configuration:** split years, row caps, context sizes, strategies, seeds, the scoring
thresholds, the prior's default-rate envelope, the arms an experiment scores.

**Deliberately in code:** the `CENSORED = -1` sentinel and the observation-mask semantics
(part of the data contract — a configurable sentinel silently reinterprets every stored period
label); numerical guards such as `_SCALE_FLOOR` and the hazard clamp, whose values follow from
float32 behaviour rather than from a modelling choice; the accounting identities in the prior,
which are arithmetic; and dataset URLs and licence attributions, which are provenance.

The financial prior's rate constants are a middle case worth noting: the **values** are
configurable, but they stay declared in `prior/financial.py` with the measured reasoning for
each, and configuration overrides them by argument. Moving the numbers into YAML would have
separated them from the paragraphs explaining why they are what they are.

## A misspelled key is an error, not a default

`load_config` refuses unknown keys and names the offending path:

```
unknown config key(s) v4finbench.train_untill; valid keys here are hazard_arms, max_context, ...
```

A silently-tolerant loader is the worst kind: the run completes, reports numbers, and used the
default. This project has already lost a pretraining run to a value that was quietly not what
it appeared to be (`docs/results/FINDINGS.md` §28), so validation happens before a run starts —
including refusing an overlapping train/test split, which would be reported as out-of-time
validation while not being it.

## Published recipes pin every key

`configs/published/` holds the training recipe of each published checkpoint, and inverts the
rule above on purpose: it states every `model`, `prior` and `train` key `fintfm-train` reads,
leaving only the per-invocation ones (`device`, `run_steps`, `resume`, `init_from`) to the
command line. An override that inherits a key moves when the packaged default moves, which is
right for an experiment and wrong for a record of what produced a set of weights.

```bash
uv run fintfm-train --config configs/published/v4-cellattn-labels.yaml --device cuda \
    --out runs/v4-cellattn-labels.pt
```

Every `fintfm-train` flag defaults to `None` and takes its value from these sections, so the
file alone is the recipe (design review DR1.3). `tests/test_train.py` checks that the file
pins every key and builds the published checkpoint's stored `ModelConfig`.
