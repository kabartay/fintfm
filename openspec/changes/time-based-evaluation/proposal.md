# Evaluate on time-based splits, not random ones

## Why

Every number in this repository comes from a **random** stratified split. For credit data
that is optimistic in a way that will not survive contact with a validation committee: a
random split lets the model see 2009 firms in training and 2008 firms in test, so it can
learn the crisis rather than generalise across it. The same firm can appear on both sides.

The leakage literature names this directly as its second mode — memorisation of global
patterns induced by external shocks — and supervisory out-of-time validation exists precisely
to prevent it. Since the product is validation evidence (`docs/DECISIONS.md` D3), reporting
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

- **Blocked by** the UCI panels carrying usable period information. The bankrupt firms span
  2000-2012 and the operating firms 2007-2013, but it is unverified whether per-firm dates
  are in the files. **Establish that first** — if they are not, this needs
  `second-credit-panel` to land before it can be done at all.
- **Blocks** any external claim about accuracy. Do not publish a headline AUC before this.
