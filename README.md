# FinancialTFM

A from-scratch, independently implemented prior-fitted tabular network aimed
at financial risk prediction (corporate default / credit risk first). This is
a research PoC, not a product: it validates whether an in-context tabular
model — no gradient steps on the customer's own data — can compete with
gradient-boosted trees on financial tables.

## How it works

1. **Priors** (`src/fintfm/prior/`): synthetic task generators.
   - `financial.py` — a structural generative story for company balance
     sheets / P&L / default labels (leverage, coverage, margin, macro cycle,
     sector effects, missingness, redundant/noise columns).
   - `scm.py` — a generic random-MLP structural-causal-model prior (the
     TabPFN-style idea), for general nonlinear/multiclass structure.
   - `mixture.py` — samples from both and batches them for training.
2. **Model** (`src/fintfm/model.py`): a from-scratch Transformer in three
   stages — cells are embedded individually, attend across *columns* within a
   row, then pool into one vector per row before rows attend to context rows.
   Predictions are provably invariant to column order and to padding width
   (both asserted in tests), which a flat row-vector design cannot be.
   Independent implementation of the alternating row/column attention idea in
   the public TabPFN/TabICL/TabFM literature — no code or weights from any
   existing project.
3. **Training** (`src/fintfm/train.py`): infinite synthetic data, one
   gradient step per fresh batch, cosine LR schedule.
4. **Inference** (`src/fintfm/classifier.py`): an sklearn-compatible
   `FinancialTFMClassifier` — `fit()` just stores the table as context,
   `predict_proba()` runs the frozen network.
5. **Metrics** (`src/fintfm/metrics.py`): AUC alone is rank-only and cannot
   see whether a stated 2% probability of default happens 2% of the time —
   the number a lender actually prices against. Every result also reports
   Brier score, expected calibration error, reliability bins, and recall at a
   base-rate operating point.
6. **Benchmark** (`src/fintfm/bench.py`): compares against logistic
   regression, random forest, gradient boosting and (if installed) LightGBM,
   on held-out synthetic tasks, real OpenML datasets, and real corporate
   defaults (`--credit`, UCI Polish bankruptcy, CC-BY-4.0, evaluation only).

## Quickstart

```bash
uv sync --extra bench
uv run fintfm-train --steps 5000 --d-model 128 --n-layers 4 --max-features 64 \
    --max-classes 2 --out runs/v0-smoke.pt
uv run fintfm-bench --model runs/v0-smoke.pt --credit      # real corporate defaults
uv run fintfm-bench --model runs/v0-smoke.pt --synthetic-tasks 10
uv run pytest
```

A real pretraining run (tens of thousands of steps, larger `--d-model`)
needs a GPU; `--device cuda` or `--device mps` on Apple Silicon. Don't run
one on this machine without checking `uptime` first — see `CLAUDE.md`.

## Configuration

The numbers experiments use — split years, context sizes, strategies, seeds, scoring
thresholds, the prior's default-rate envelope — live in
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
belongs in configuration and what deliberately stays in code.

## Licensing / provenance

Everything here — the prior, the architecture, the training loop — is
original code written for this project. No weights, datasets, or source
from Neuralk (Seldon), Fundamental (NEXUS), Google TabFM, TabPFN/TabICL,
or any other tabular-foundation-model product are used. Those are cited in
project discussion purely as public research/product context, not as a
source of code or data. Before adding any third-party dataset or dependency,
check its license against the intended commercial use. Current dependencies are all
permissive: numpy, pandas and scikit-learn BSD-3, scipy BSD-3, torch Apache-2.0,
**PyYAML MIT**, and the optional benchmark extras lightgbm MIT, xgboost Apache-2.0,
catboost Apache-2.0, pyarrow Apache-2.0.

## Where this is going

[`docs/`](docs/) holds the project's reasoning, indexed in
[docs/README.md](docs/README.md). Start with [STRATEGY.md](docs/STRATEGY.md)
for the plan of record, [ARCHITECTURE.md](docs/ARCHITECTURE.md) for how the
model works, [DECISIONS.md](docs/DECISIONS.md) for why it is built this way and
what would reverse each choice, and [FINDINGS.md](docs/FINDINGS.md) for measured
results.

Short version: the mechanism claims ("no training on your data", "no feature
engineering") are already owned by better-funded competitors, so the thesis is
not the mechanism. It is a credit model that arrives with its own validation
evidence — calibrated, auditably free of benchmark contamination, and eventually
backed by a pre-registered forward track record that cannot be bought.

## Status

PoC skeleton: priors + architecture + training + benchmark harness are
implemented, unit-tested (10 tests, `uv run pytest`), and verified to run
end-to-end on CPU (300-step, 172K-parameter smoke run; checkpoint not kept).
That smoke run is a **pipeline check only** — it says the training loop
converges on the synthetic prior and the benchmark harness produces sane,
non-`NaN` numbers, not that the architecture is any good. Its AUC on
held-out synthetic tasks was roughly at parity with the classical baselines,
which is the expected result of a model 100–1000x smaller and shorter than
a real pretraining run, not a finding.

Not yet done, in order: a real pretraining run (needs GPU hours this
environment doesn't have — see `CLAUDE.md`), benchmarking against public
credit-risk datasets (e.g. a V4FinBench-style corporate panel) with proper
time-based splits, then iterating on the financial prior based on where it
under/over-performs against gradient-boosted trees.
