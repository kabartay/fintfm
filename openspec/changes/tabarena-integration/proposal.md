# Benchmark fintfm inside TabArena, for a number comparable to published work

## Why

Every real-data number this project has is on data this project chose: V4FinBench, Polish and
Taiwan bankruptcy, and the ad-hoc 15-dataset OpenML suite in `experiments/openml_breadth.py`.
The comparators are logistic regression and gradient boosting **that we ran ourselves**. So
there is no calibrated sense of where fintfm sits against the field — only against our own
baselines, on our own splits.

TabArena (`github.com/autogluon/tabarena`, **Apache-2.0**) fixes exactly that. Reconnaissance
2026-09-19:

* **TabArena-v0.1**: 51 curated datasets, 9-30 splits each.
* **BeyondArena**: 142 datasets spanning **IID, temporal and grouped** tasks, tiny to **1M
  rows**. That is V4FinBench's own shape — company-grouped, temporally split, ~1M rows — and
  it is the first source of *non-credit* temporal tasks available to this project.
* Published leaderboard numbers for TabPFN v3, TabICL v2, Mitra and tuned tree ensembles, so
  the field's position comes for free rather than from re-running competitors.

## What changes

A fintfm model folder in a **local clone**, not vendored into this repository: TabArena's
`[benchmark]` extra pulls AutoGluon and a large dependency tree, and this project's
deliberately minimal, uniformly-permissive dependency surface is a licensing asset worth
keeping (`README.md`, licensing section).

Integration shape, from `.claude/skills/add-model/SKILL.md` and the TabICL reference:

| file | contents |
| --- | --- |
| `models/fintfm/model.py` | an `AbstractTorchModel` subclass: `get_model_cls`, `_fit`, `_estimate_memory_usage_static`, `_set_default_params` |
| `models/fintfm/hpo.py` | a `ConfigGenerator` search space |
| `models/fintfm/info.py` | `MethodMetadata` — `license="Apache-2.0"`, `compute="gpu"`, `can_hpo` |

**`_supported_problem_types` must be `["binary"]`.** Current checkpoints are `max_classes=2`
and `max_features=136`; declaring honestly makes the harness *skip* what fintfm cannot do
rather than score it wrongly. Coverage is then reported as "ran N of M", which is the number
that matters.

## Non-goals

- **Not a leaderboard submission.** Submission requires evaluating on TabArena-Lite, opening a
  PR, and maintainers re-running for verification. With binary-only, <=136-feature coverage
  and a measured ~0.11-0.20 AP deficit to *default* gradient boosting
  (`experiments/openml_breadth.py`, §96), a submission would be partial coverage and last
  place, permanently recorded, for no information this project does not already have.
- **Not adopting their preprocessing as ours.** TabArena applies its own shared preprocessing
  and validation protocol. That is what makes numbers comparable and it also means a TabArena
  score is not the same measurement as a V4FinBench score; the two must not be quoted
  interchangeably.
- **Not a substitute for the credit panels.** The product claims — coherent PD term structures,
  calibration, auditable provenance — are not measured by TabArena at all. A leaderboard would
  steer this project toward general tabular accuracy, the one axis where three funded teams
  are already at parity with each other.

## Blocked by

- **The fin00 breadth control** (`experiments/openml_breadth.py` on the `p_financial=0.0`
  checkpoint). §96's suite ran a **financial-prior-only** checkpoint on ecology, physics,
  chemistry and telecom data, so its "broadly weak" reading is confounded with prior coverage.
  If the SCM-prior checkpoint closes most of the gap, the next measurement is a prior
  experiment, not a benchmark, and this proposal drops in priority.
- **Machine load.** The AutoGluon install and a Lite run compete with local scoring; do not
  start either while a V4FinBench run is in flight.
