# Evaluate on a diverse tabular suite, not three credit panels

## Why

`docs/results/FINDINGS.md` §44 located the defect: the model reaches held-out AUC 0.83-0.88 with
positive Brier skill on tasks from its own prior, and collapses to 0.685 on iid Gaussian
features. **It learns in context; it does not generalise across feature distributions.**

Our evaluation cannot see that. It is three credit panels — Polish, Taiwanese, V4FinBench —
plus synthetic probes, all narrow, and one of them (V4FinBench horizon 0) is nearly solved by
a single column (§42). A model that fits one family of feature distributions and nothing else
scores well on all of it. **This is the instrument that would have caught §42 on day one
rather than day three.**

`inria-soda/tabular-benchmark` on Hugging Face is the obvious candidate: the Grinsztajn et al.
suite behind *"Why do tree-based models still outperform deep learning on tabular data?"*,
which is the exact question this project is a bet against. 23 classification datasets — 7 with
categoricals, 16 numeric-only — and 36 regression, as plain CSV. It also contains
`default-of-credit-card-clients`, a set we already evaluate on, which gives a calibration
point between the two suites.

## What

- Resolve licences **first**, per dataset, and record them. The HF card declares **no licence**
  at the collection level; the underlying sets come from OpenML and UCI with individual terms,
  so "it is a public benchmark" is not a licence (`CLAUDE.md`).
- Add a loader for the classification subset and score every arm on all of it: our checkpoints,
  the untrained control, logistic regression and the boosters.
- Report **native coverage** alongside accuracy — the fraction of datasets scored without
  falling back — per `adopt-published-methods` task 36.4.
- Report the *spread* across datasets, not only the mean. A model that generalises has a
  narrow spread; one that fits a single distribution family has a wide one, and the mean hides
  exactly the failure §44 describes.

## Non-goals

- **Not pretraining data.** Decision D2 is unaffected and unnegotiable: a model that never saw
  real data cannot have memorised a benchmark, and that auditability is the product. These sets
  are for evaluation only, and the distinction is now explicit in `CLAUDE.md`'s licensing
  boundary.
- Not a claim to compete on general tabular accuracy (D3). This is an instrument for measuring
  generalisation, not a leaderboard entry.

## Falsified by

Nothing — it is a measurement. The informative outcome is the *spread*: if our model's per-
dataset scores are wide while the boosters' are narrow, §44's generalisation diagnosis is
confirmed on real data rather than on synthetic probes alone.

## Blocked by

Licence resolution (task 37.1). Nothing else — this needs no training run and would have value
even against the current, weak checkpoints, because it measures the shape of the failure.
