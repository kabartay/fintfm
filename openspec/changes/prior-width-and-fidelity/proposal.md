# Close the width and fidelity gap between the prior and real credit panels

## Why

The prior is the one component this project owns, and `docs/results/FINDINGS.md` §18 measured exactly
how it diverges from the data it must transfer to:

- **Width: 9-21 features generated against 64-95 in real panels.** The generator is
  hard-capped near 24 columns by a fixed dictionary of ~20 quantities, *regardless of
  `max_features`*. This is a 3-5× mismatch on the axis the architecture is most sensitive to,
  and the most plausible remaining explanation for the modest +0.033 AUC over logistic
  regression (§17).
- **Correlation: 0.096-0.220 mean absolute inter-feature correlation against 0.081-0.110
  real**, because exposed features derive from a handful of latents.

Difficulty, by contrast, is already well matched (logistic-regression AUC 0.761 synthetic
against 0.769 real), so **this change is about shape, not hardness** — and it should not
break the difficulty match, which must be re-measured to confirm.

## What

- **Derive a large ratio family from the existing latent accounts.** Real credit datasets are
  wide precisely because they compute many ratios over one balance sheet and P&L; the Polish
  panel's 64 features are of that kind. The prior already generates the accounts, so this is
  faithful rather than synthetic padding: profitability, liquidity, leverage, coverage,
  turnover, working-capital and growth ratios, with the numerators and denominators sampled
  from the account set.
- **Sample the exposed width up to `max_features`**, so a task can genuinely be 64 or 95
  columns wide and the configuration means what it says.
- **Re-measure all four properties in §18 afterwards** — width, difficulty, correlation and
  missingness — and treat a lost difficulty match as a regression.
- **Trim the easy tail.** Tasks at logistic-regression AUC 0.98 teach little; consider
  rejecting or down-weighting near-separable draws.

## Non-goals

- Not fitting any distribution to a real panel. That would void the provenance invariant
  (`openspec/specs/pretraining-provenance` P2) and is the one thing this change must not do,
  however tempting when matching a correlation statistic.
- Not adding temporal structure; that is `temporal-financial-prior`.
- Not adding hazard paths; that is `pd-term-structure`.

## Falsified by

Retrain at matched compute with the wider prior and compare on the real panels. If AUC does
not improve over the current prior, width was not the binding constraint and §18's leading
explanation is wrong — which is worth knowing, since the alternative explanations (the
architecture is near its ceiling, or 2.2M parameters is simply too small) point somewhere
quite different.

## Blocked by / blocks

- **Blocked by** nothing. Independent of everything else in the queue.
- **Blocks** a fair test of the architecture's ceiling, because every result so far was
  obtained with a prior 3-5× narrower than the evaluation data.
