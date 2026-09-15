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

## What is currently true, in three sentences

A synthetic hazard-head architecture makes incoherent PD term structures (a real, measured
defect in the field's standard per-horizon construction, 39% of firms on real data) impossible
by construction, at zero accuracy cost. A severe capacity defect in the original architecture —
capped discrimination regardless of true task difficulty, verified against an exactly-known
Bayes-optimal AUC — was found, its cause isolated to the architecture rather than the prior
after eliminating seven other candidates one at a time, closed by a two-way cell-attention
change, and confirmed on real data at matched context (+0.049 average precision on
V4FinBench, five folds of five) — though that architecture's memory cost currently puts the
project's best-scoring inference configuration out of reach. On real credit panels, this project's
calibration is consistently among the best measured, and its discrimination consistently loses
to tuned gradient boosting — both facts, together, on every panel tried.

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
5. **Evaluation** (`src/fintfm/evaluation/`, `src/fintfm/experiments/`) — real corporate-default
   panels (V4FinBench via its published protocol, `fintfm-v4protocol`; Polish and Taiwan
   bankruptcy, `fintfm-bench`), an out-of-time harness, a synthetic capability suite
   (`fintfm-capability`) including probes with a *known* ceiling — some with a closed-form
   Bayes-optimal AUC — specifically built to catch a model that looks fine on average while
   capped in a way an aggregate score cannot see. Average precision is reported alongside
   ROC-AUC everywhere, and read first at low base rates (ROC-AUC's chance floor is 0.5
   regardless of prevalence; AP's floor is the prevalence itself, so it stays legible at the
   base rates this project's target segment actually has).

## Quickstart

```bash
uv sync --extra bench
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
  (80 entries and counting), each declaring how its numbers were produced.
- [`docs/DECISIONS.md`](docs/DECISIONS.md) — why the project is built the way it is, and what
  would reverse each choice.
- [`docs/STRATEGY.md`](docs/STRATEGY.md) — the plan of record.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — how the model works, in more depth than
  this file.
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

## Status

Actively developed research codebase, not a PoC skeleton: 184 tests (`uv run pytest`), a
config-driven experiment harness, real GPU pretraining infrastructure (Hugging Face Jobs on
T4), and 80 numbered, provenance-tagged findings. What is currently proven, currently open, and
currently retracted is tracked continuously in
[`docs/paper/CLAIMS.md`](docs/paper/CLAIMS.md) rather than restated here, because the honest
state changes faster than this file gets edited — that is exactly the failure mode the claims
ledger exists to prevent.
