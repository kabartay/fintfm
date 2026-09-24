# Re-measure context strategy on the binary path

## Why

`docs/results/FINDINGS.md` §29 measured uniform context sampling beating balanced by **10-12 mean AUC
points** on the V4FinBench out-of-time survival split, at every context size tested, with the
ordering uniform > hybrid > balanced replicated three times. That is the *reverse* of the
published result the class default was adopted from (Tanna et al. 2026: balanced worth 3-4 AUC
points), and three times the magnitude.

Decision D9 deliberately did **not** flip the class default, because §29 measured only the
six-horizon survival path and the binary single-horizon evidence behind D5 has never been
re-measured under uniform sampling. So the repository currently ships a default that one
experiment contradicts and no experiment supports on its own path. That is an unstable state
and it should be resolved by measurement rather than left to whichever finding is read last.

The mechanism §29 proposes is testable and, if true, general: balancing a low-default portfolio
spends most of the context budget on a *thin sample of the majority class* (1,000 of 71,500
non-defaulters), and in a low-default portfolio that is where the decision boundary lives. If
that is the mechanism, the effect should appear on the binary path too, and should shrink as
the portfolio's default rate rises.

## What

- Sweep `{uniform, hybrid, balanced}` × context size on the binary path across every available
  panel (Polish, Taiwan, Home Credit, V4FinBench single-horizon), with base-rate correction on
  throughout so §28's confound cannot recur.
- Multiple seeds per cell, with paired bootstrap and Holm-Bonferroni correction — §29 is a
  single draw per cell and must not be promoted on that basis.
- Test the proposed mechanism directly: regress the uniform-minus-balanced AUC gap on the
  portfolio default rate. The mechanism predicts the gap shrinks as the rate rises and vanishes
  near balance.

## Non-goals

- Not changing the class default until this returns. D9 holds until then.
- Not re-litigating Tanna et al. on their setting. Their benchmark stands; the question is which
  regime each result applies to.

## Falsified by

If uniform loses on the binary path, or if the gap fails to track the default rate, then §29's
mechanism is wrong and its result is specific to the survival path — which would itself be
worth knowing, and would make the split default in D9 correct rather than provisional.

## Blocked by

Nothing for tasks 32.1-32.4. Task 32.5's default change is blocked on 32.3 returning
Holm-corrected significance, since §29 is one draw per cell and a class default must not turn
on a single seed.
