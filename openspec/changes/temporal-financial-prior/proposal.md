# Give the financial prior a time axis

## Why

The prior generates a **single cross-sectional snapshot** per company: one balance sheet, one
P&L, one label. Real credit data is a *panel* — the same firm observed over several years —
and the thing a lender actually asks is "will this firm default within N years, given its
trajectory". Deterioration is the signal: falling margin with rising leverage over three
years is a different risk from the same ratios held steady.

The UCI panels already expose this and we are throwing it away. The dataset ships five files,
one per forecast horizon, and the current loader treats them as five unrelated datasets
rather than one panel with five horizons.

This is the largest known gap between the prior and the task it is meant to prepare for.

## What

- Extend `prior/financial.py` to generate a multi-year trajectory per firm: persistent
  company effects, autocorrelated ratios, a macro regime that evolves rather than being drawn
  once, and refinancing events at debt maturity.
- Emit trajectory-derived features the way a real panel would expose them: levels, deltas,
  multi-year trends, volatility.
- Label default within a horizon that is itself sampled, so the model learns horizon as a
  property of the task rather than a constant.

## Non-goals

- **Not** becoming a time-series model. Rows stay exchangeable and the architecture stays
  tabular; the trajectory is encoded as features, not as a sequence the model attends over.
  The Forecasting Company's argument that time series are not tables is correct, and this
  proposal deliberately stays on the tabular side of that line.
- Not fitting trajectory dynamics to a real panel — that would break the parametric-prior
  invariant (`docs/design/ARCHITECTURE.md`).

## Falsified by

Cheapest test: build trajectory features from the existing UCI panels by joining the five
horizon files per firm, and check whether gradient boosting improves when given deltas and
trends versus levels alone. If the trajectory carries no signal on real data, generating it
synthetically cannot help either.

## Blocked by / blocks

- **Blocked by** `phase1-prior-ablation` — do not enrich the prior before knowing whether the
  prior transfers at all.
- **Blocks** nothing, but it is the most likely single source of a real accuracy gain.
