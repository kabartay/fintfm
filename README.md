# fintfm

A from-scratch, independently implemented tabular foundation model — an in-context classifier
pretrained on synthetic tasks, aimed first at corporate credit risk. This is a research
project with a public claims ledger, not a product: it exists to find out whether in-context
tabular learning (no gradient steps on a customer's own data) can be made competitive with
gradient-boosted trees on financial tables, and to say honestly where it currently is not.

**Read [`docs/paper/CLAIMS.md`](docs/paper/CLAIMS.md) for the current state of every claim
this project could make, each tagged SURVIVES / SINGLE DRAW / SUPERSEDED / RETRACTED / OPEN
against its evidence.** That file, not this one, is the source of truth for what is currently
true. This README is an orientation map.

## What is currently true

**On real credit panels, calibration is consistently among the best measured and discrimination
consistently loses to tuned gradient boosting** — both facts together, on every panel tried.
That is the project in one line, and everything below is detail.

**Two things it does that nothing else does.** A hazard head makes incoherent PD term
structures impossible by construction — a real, measured defect in the field's standard
per-horizon construction, affecting 39% of firms on real data — at zero accuracy cost. And a
severe capacity defect in the original architecture (capped discrimination regardless of true
task difficulty, caught against an exactly-known Bayes-optimal AUC) was isolated to the
architecture rather than the prior after eliminating seven other candidates one at a time, then
closed by two-way cell attention and confirmed on real data at +0.042 average precision.

**Measured externally on [TabArena](docs/TABARENA.md), it places 93rd of 95** (§98), and the
diagnosis of that deficit is the most useful work here. Most of it was *preprocessing*, not
architecture: the per-dataset gap correlated **−0.668** with log categorical cardinality, and
replacing label encoding with out-of-fold target statistics lifted the mean from 0.7642 to
**0.7823** while collapsing that correlation to −0.025 (§100, §101). **The rank did not move.**
What remains is a uniform ~0.035 ROC-AUC deficit with no measured axis of variation — a sharper
target than before, and still a losing one.

**What has since been ruled out, which is most of what looked promising.** `column_id_dim`, the
only untuned lever that had ever moved real-data accuracy, peaks at the value chosen by accident
(§104). Parameter scale lost (§108). Training volume at 5× was inert (§93). Class balance did
not survive testing (§102, §103). A widened SCM prior was a flat null against its own control
(p = 1.000). Reading the field explains why: a peer's published scaling curve returns +0.005 R²
for 16× the parameters, so **no scaling programme was ever going to close 0.035** — and every
one of TabArena's top fourteen ranks is synthetic-pretrained (§110), which says the constraint
this project chose costs nothing in rank and the lever is prior *design*.

**Coverage, which any score must carry.** The 93rd place was measured at **51% coverage**;
multiclass and regression have since taken it to **90% (46 of 51)**, with `max_features` the
only remaining exclusion. Those arms are *runnable*, not yet *scored* on real data, and a
coverage fraction is not a result.

**The honest one-line summary: this is not a competitive general tabular model, and on TabArena
its median rank is 94 of 95 — including on the credit panels.** An earlier version of this line
claimed its best public results were the corporate-credit panels it was designed for. §107
retracts that: those panels rank 83–95, and the high absolute AUC on them (0.9287 on Taiwanese
bankruptcy) is what everyone scores there, not an edge. Whether a credit specialism exists is a
question for V4FinBench's five-fold protocol, not for a single-fold leaderboard.

## How it works

1. **Priors** (`src/fintfm/prior/`) — synthetic task generators, sampled and mixed per batch
   (`mixture.py`).
   - `financial.py` — a structural generative story for company balance sheets, P&L and
     default labels: accounting identities, sector and macro effects, a wide derived-ratio
     family, missingness, a sharpness (signal-to-noise) knob, and a default-rate envelope that
     can reach real low-default-portfolio rates (~0.2%).
   - `scm.py` — a generic random-graph structural-causal-model prior (the TabPFN/TabICL-style
     idea): random layered functions with several nonlinearities, for general nonlinear and
     multiclass structure the financial prior does not cover.
   - `tree.py` — ensembles of oblivious decision trees, generating the **axis-aligned,
     piecewise-constant** structure real tabular data is full of and the other priors do not.
     Added on *distinctiveness* grounds after `fintfm-priorscore` measured it as the only
     member of the mixture with a positive tree-versus-linear gap (§112). Off by default
     (`p_tree=0.0`).
   - `trivial.py`, `crossed.py` — diagnostic-only priors used to isolate specific hypotheses
     (whether the architecture can learn at all; whether a prior's features or its label
     mechanism carries a measured effect). Not part of the default training mixture.
2. **Model** (`src/fintfm/modeling/model.py`) — a from-scratch Transformer. Cells are embedded
   individually and carry a random per-task column identity (so the model can tell columns
   apart without a positional embedding, which would break invariance to column order).
   Columns attend to each other within a row; optionally — `ModelConfig.n_cell_blocks`,
   currently experimental — cells also attend across rows *within one feature* before that,
   giving each column a distribution-derived identity rather than only a random tag. Rows are
   then pooled and attend to context rows to perform in-context learning. An optional hazard
   head (`modeling/hazard.py`) produces a cumulative PD term structure that is monotone by
   construction. Independent implementation of ideas described in the public TabPFN / TabICL
   literature — no code or weights from any existing project (see Licensing below).
3. **Training** (`src/fintfm/modeling/train.py`, `fintfm-train`) — infinite synthetic data, one
   gradient step per fresh batch, cosine LR schedule, checkpointing, a held-out quality metric
   that reports per-task AUC (not the pooled-across-tasks number, which is inflated whenever
   task base rates differ) and Brier skill against a base-rate-only predictor.
4. **Inference** (`src/fintfm/inference/`) — an sklearn-compatible `FinancialTFMClassifier`:
   `fit()` stores the table as context, `predict_proba()` runs the frozen network. Several
   context-construction strategies (uniform, balanced, hybrid, retrieval, prototype) and a
   base-rate correction, plus ensembling over column-identity draws to recover the
   distributional column-order invariance a random identity trades for expressiveness.
   `inference/categorical.py` encodes categorical columns as **out-of-fold** smoothed target
   statistics, because the model reads every cell as an ordered scalar and label encoding's
   arbitrary order is measurably worse than no order at all (§100). The out-of-fold step is
   not a refinement: the naive form puts a row's own label into its own encoding, which makes
   the context self-predictive and the feature absent at query time — it harms the model
   rather than flattering the score, so it survives careless validation.
   `inference/regressor.py` adds a `FinancialTFMRegressor` on the same frozen network and no
   new architecture: a continuous target cut into K quantile bins is an integer index over K
   outcomes, so the existing classification head regresses as-is, and the output is natively
   **distributional** — quantiles and prediction intervals come free, and the predicted
   density can be bimodal, which is what loss given default actually is and what a Gaussian
   head cannot represent.
5. **Evaluation** (`src/fintfm/evaluation/`, `src/fintfm/experiments/`) — real corporate-default
   panels (V4FinBench via its published protocol, `fintfm-v4protocol`; Polish and Taiwan
   bankruptcy, `fintfm-bench`), an out-of-time harness, a synthetic capability suite
   (`fintfm-capability`, including a class-count sweep for multiclass, §99) with probes of a
   *known* ceiling — some with a closed-form
   Bayes-optimal AUC — specifically built to catch a model that looks fine on average while
   capped in a way an aggregate score cannot see. Average precision is reported alongside
   ROC-AUC everywhere, and read first at low base rates (ROC-AUC's chance floor is 0.5
   regardless of prevalence; AP's floor is the prevalence itself, so it stays legible at the
   base rates this project's target segment actually has).
   `fintfm-priorscore` scores a *prior* rather than a model, on the three criteria the
   literature converged on — performance, diversity, distinctiveness — using **fitted**
   baselines only, so it cannot confuse "the prior lacks this structure" with "our model cannot
   learn it". Its first version could, and §112 is the record of what that cost.

## Quickstart

```bash
uv sync --extra bench --extra hf   # naming one extra uninstalls the others
uv run fintfm-train --steps 300 --d-model 32 --d-cell 16 --n-layers 2 --max-features 16 \
    --max-classes 2 --device cpu --out runs/v0-smoke.pt   # pipeline check, a couple of
                                                            # minutes, says nothing about quality
uv run fintfm-bench --model runs/v0-smoke.pt --credit      # real corporate-default panels
uv run pytest
```

That smoke config exists to check the pipeline runs, not to produce a usable checkpoint — see
[`docs/COMPUTE.md`](docs/COMPUTE.md) for measured step costs at real scale. A real pretraining
run (thousands of steps, `--d-model 128`+, `--max-features` matching your data) takes
hours-to-a-day and needs a GPU: `--device mps` on Apple Silicon, `--device cuda`, or see
[`docs/HF_JOBS.md`](docs/HF_JOBS.md) for the Hugging Face Jobs recipe this project actually
uses for training runs. **Check `uptime` before running anything heavy locally** — see
[`CLAUDE.md`](CLAUDE.md) for why.

Both `fintfm-bench --credit` (Polish/Taiwan need 64/95 features) and `fintfm-v4protocol`
(V4FinBench needs 136) will print `SKIPPED for fintfm: model takes 16 features, data has N` —
correct, expected behaviour for the smoke checkpoint above, not a bug. Baselines still run and
score normally; only the fintfm arm needs a checkpoint pretrained with a matching
`--max-features` to be evaluated.

## Configuration

The numbers experiments use — split years, context sizes, strategies, seeds, scoring
thresholds, the prior's default-rate and sharpness envelope — live in
[`src/fintfm/configs/default.yaml`](src/fintfm/configs/default.yaml), not scattered through
the code. Every entry point takes `--config` with a file that is **deep-merged** over that
default, so an override carries only what it changes:

```bash
uv run fintfm-ctxsweep --model runs/m.pt --config configs/context-sweep-3seed.yaml
uv run fintfm-v4oot    --model runs/m.pt --config configs/retrieval-best.yaml
```

Resolution order is packaged default → `--config` (or `$FINTFM_CONFIG`) → explicit CLI flag.
Runs print the layers they used and record them as `config_sources` in their output JSON.
Unknown keys are refused by name rather than ignored, and an overlapping train/test split is
rejected before the run starts. [`configs/README.md`](configs/README.md) explains what
belongs in configuration and what deliberately stays in code; several files under
[`configs/`](configs/) are **deliberately diagnostic, not production** (their own headers say
so), used to isolate a single variable while chasing a specific finding.

## Licensing / provenance

Everything here — the prior, the architecture, the training loop — is original code written
for this project. **No weights, datasets, or source from Neuralk (Seldon), Fundamental
(NEXUS), Google TabFM, TabPFN, TabICL, TabDPT, or any other tabular-foundation-model product
are used or may be added.** Those are cited in project discussion purely as public
research/product context. Every third-party dataset or dependency's licence is checked before
use, and a *weight* licence is checked separately from its *code* licence, every time — never
assumed from a prior check. Current dependencies are all permissive: numpy, pandas and
scikit-learn BSD-3, scipy BSD-3, torch Apache-2.0, PyYAML MIT, and the optional benchmark
extras lightgbm MIT, xgboost Apache-2.0, catboost Apache-2.0, pyarrow Apache-2.0.

## Where this is going

[`docs/`](docs/) holds the project's reasoning, indexed in [`docs/README.md`](docs/README.md).

- [`docs/paper/CLAIMS.md`](docs/paper/CLAIMS.md) — **start here.** Every claim this project
  could make, tagged by status, newest evidence wins.
- [`docs/FINDINGS.md`](docs/FINDINGS.md) — the full measurement log, numbered sequentially
  (113 entries and counting), each declaring how its numbers were produced.
- [`docs/DECISIONS.md`](docs/DECISIONS.md) — why the project is built the way it is, and what
  would reverse each choice.
- [`docs/STRATEGY.md`](docs/STRATEGY.md) — the plan of record.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — how the model works, in more depth than
  this file.
- [`docs/TABARENA.md`](docs/TABARENA.md) — the external-evaluation recipe, the coverage
  fraction any score must carry, and four silent failure modes including a results cache that
  returns stale numbers after a preprocessing change.
- [`docs/paper/RELATED_WORK.md`](docs/paper/RELATED_WORK.md) — the competing models, read from
  primary sources, with what each costs this project's positioning. Its digest is the shortest
  useful summary of why the remaining lever is the prior.
- [`docs/POSTMORTEM.md`](docs/POSTMORTEM.md) — wrong diagnoses, each caught by measurement
  rather than review, kept on the record deliberately.
- [`openspec/changes/`](openspec/changes/) — the live roadmap as structured proposals with
  verifiable tasks, not prose. `uv run python openspec/tools/validate.py` checks every
  proposal is well-formed and every finding declares its provenance.

Short version: the mechanism claims ("no training on your data", "no feature engineering") are
already owned by better-funded competitors, so the thesis is not the mechanism. It is a credit
model that arrives with its own validation evidence — calibrated, auditably free of benchmark
contamination, and eventually backed by a pre-registered forward track record that cannot be
bought — plus a public, self-correcting record of what has and has not been shown to be true.

## What would change the picture

Stated so the project is falsifiable rather than open-ended. The uniform ~0.035 deficit has no
measured axis; the levers that remain, in the order the evidence ranks them:

| lever | status |
| --- | --- |
| **prior design** — a tree-structured prior, measured as the only distinctive member of the mixture | arms trained, scoring |
| **factorized attention** — the current encoder is memory-bound at every turn, and three peers independently chose the cheaper form | proposed (44.x), prior art recorded |
| **objective** — `p(x, y \| D)` rather than `p(y \| x, D)`, which makes every column a training signal | proposed (48.15), scoped as a measurement before a rewrite |
| ~~parameter scale~~ | ruled out: a peer's curve returns +0.005 R² for 16× parameters |
| ~~`column_id_dim`~~ | closed (§104) |
| ~~training volume at 5×~~ | null (§93) |

## Status

Actively developed research codebase, not a PoC skeleton: 222 tests (`uv run pytest`), a
config-driven experiment harness, real GPU pretraining infrastructure (Hugging Face Jobs on
T4), an external benchmark integration (TabArena, `docs/TABARENA.md`, 90% coverage), and 113
numbered, provenance-tagged findings. What is currently proven, currently open, and
currently retracted is tracked continuously in
[`docs/paper/CLAIMS.md`](docs/paper/CLAIMS.md) rather than restated here, because the honest
state changes faster than this file gets edited — that is exactly the failure mode the claims
ledger exists to prevent.
