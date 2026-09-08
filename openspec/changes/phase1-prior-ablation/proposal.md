# Does a financial prior beat a generic one at matched compute?

## Why

This is the experiment that decides whether the project has a reason to exist. The whole
thesis rests on domain-specific synthetic pretraining transferring to real credit tables. If
a generic structural-causal prior does just as well, then the financial prior — the one
component competitors are not building — buys nothing, and the honest response is to say so
publicly and fall back to the validation layer, which needs no model of our own.

Nothing measured so far speaks to this. Every number to date came from smoke-scale
checkpoints whose job was to prove the pipeline runs.

## What

`fintfm-ablate` trains one model per prior mixture — `financial` (p=1.0), `mixed` (p=0.7),
`generic` (p=0.0) — holding architecture, parameter count, optimiser, steps, batch size,
seed and evaluation identical, then scores all three on the same paired splits of the real
corporate-default panels with calibration as well as ranking.

The harness refuses to report if parameter counts diverge, and writes `results.json` with
the config and git commit so every number is re-derivable.

## Exit condition, stated before the run

The `financial` variant must beat `generic` on real credit data, on calibration as well as
AUC. Recorded in the module docstring so it cannot be moved after seeing results.

## Non-goals

- Not reaching the 10-50M parameter target. The first pass is 2.2M, deliberately below it:
  find out whether the effect exists before spending a weekend measuring it precisely.
- Not beating gradient boosting. That is a separate, later question.
- Not settling balanced-versus-hybrid context strategy; that needs multiple seeds.

## Falsified by

Itself. That is the point — a null result here is the cheapest possible way to learn the
central bet is wrong.

## Blocked by / blocks

- **Blocked by** nothing. Harness built and tested; runs on Metal locally.
- **Blocks** `scaling-curve` (no point plotting a curve for an effect that does not exist),
  and everything in Phase 2+ of `docs/STRATEGY.md`.
