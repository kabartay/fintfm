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
2. **Model** (`src/fintfm/model.py`): a from-scratch Transformer encoder.
   Rows are tokens; context rows get a label embedding, query rows get a
   learned "unknown" token; attention is masked so queries never attend to
   each other, only to context. Independent implementation of the PFN idea
   described in the public TabPFN/TabICL literature — no code or weights
   from any existing project.
3. **Training** (`src/fintfm/train.py`): infinite synthetic data, one
   gradient step per fresh batch, cosine LR schedule.
4. **Inference** (`src/fintfm/classifier.py`): an sklearn-compatible
   `FinancialTFMClassifier` — `fit()` just stores the table as context,
   `predict_proba()` runs the frozen network.
5. **Benchmark** (`src/fintfm/bench.py`): compares against logistic
   regression, random forest, gradient boosting, and (if installed)
   LightGBM, both on held-out synthetic financial tasks and on real OpenML
   datasets.

## Quickstart

```bash
uv sync --extra bench
uv run fintfm-train --steps 5000 --d-model 128 --n-layers 4 --out runs/v0-smoke.pt
uv run fintfm-bench --model runs/v0-smoke.pt --synthetic-tasks 10
uv run pytest
```

A real pretraining run (tens of thousands of steps, larger `--d-model`)
needs a GPU; `--device cuda` or `--device mps` on Apple Silicon.

## Licensing / provenance

Everything here — the prior, the architecture, the training loop — is
original code written for this project. No weights, datasets, or source
from Neuralk (Seldon), Fundamental (NEXUS), Google TabFM, TabPFN/TabICL,
or any other tabular-foundation-model product are used. Those are cited in
project discussion purely as public research/product context, not as a
source of code or data. Before adding any third-party dataset or dependency,
check its license against the intended commercial use.

## Status

PoC skeleton: priors + architecture + training + benchmark harness are
implemented and unit-tested. Not yet run at meaningful pretraining scale
(that requires GPU hours this environment doesn't have). Next steps: run a
real pretraining job, benchmark against public credit-risk datasets (e.g.
V4FinBench-style corporate panels) with proper time-based splits, then
iterate on the financial prior based on where it under/over-performs.
