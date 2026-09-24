# Evaluate on time-based splits, not random ones

## Why

Every number in this repository comes from a **random** stratified split. For credit data
that is optimistic in a way that will not survive contact with a validation committee: a
random split lets the model see 2009 firms in training and 2008 firms in test, so it can
learn the crisis rather than generalise across it. The same firm can appear on both sides.

The leakage literature names this directly as its second mode — memorisation of global
patterns induced by external shocks — and supervisory out-of-time validation exists precisely
to prevent it. Since the product is validation evidence (`docs/design/DECISIONS.md` D3), reporting
random-split numbers undermines the one thing being sold.

## What

- A splitter that partitions by observation period: train on the earlier window, test on the
  later one, with no firm crossing the boundary.
- Report both random-split and time-split numbers side by side, permanently. The gap between
  them is itself a finding, and hiding it would be the failure this proposal exists to fix.
- Make the time split the headline number wherever one is quoted outside the repo.

## Non-goals

- Not rolling-origin backtesting across many cutoffs yet; that belongs with
  `forward-prediction-register`.
- Not dropping random splits. They stay as the comparison that shows how much optimism a
  random split buys.

## Falsified by

Nothing to falsify — this is a correctness fix, not a hypothesis. The open question is only
how large the gap is, and that is what the change measures.

## Blocked by / blocks

- **Blocked by `second-credit-panel` task 7.5 only.** Established 2026-09-08
  (`docs/results/FINDINGS.md` §7): the UCI panels carry no dates, periods or identifiers, so this is
  impossible on them. **Unblocked in principle by V4FinBench** (§8) — 1.1M company-year rows
  spanning 2006-2021, CC BY 4.0 — which covers the financial crisis and COVID and therefore
  supports out-of-time validation across genuine regime shift. Needs that ingest first.
- **Blocks** any external claim about accuracy. Do not publish a headline AUC before this.
