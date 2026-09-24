# Native categorical handling

## Why

`docs/results/FINDINGS.md` §100 located this project's TabArena deficit. It is not spread evenly
across the suite: on the eight datasets with no categorical columns fintfm is 0.0320 ROC-AUC
behind tuned logistic regression, and on the nine that are more than half categorical it is
0.0894 behind. Against maximum level count the pattern is sharper still — 0.0320 at zero
categoricals, 0.0437 up to 25 levels, **0.1109 above 25** — and the per-dataset gap correlates
**-0.668** with log maximum cardinality.

The cause is mechanical. `modeling/model.py` embeds every cell as a numeric scalar, so a
categorical column must arrive as a number, and §98's integration label-encoded it. That
asserts `Amazon_employee_access`'s resource code 4127 sits between 4126 and 4128 on a
meaningful axis. It does not, and that dataset scored 0.5455 against a baseline's 0.8442.

If every categorical-bearing dataset merely reached parity with tuned logistic regression, the
mean gap would fall from 0.0527 to 0.0095 — **82% of the distance to the arm ranked #89**.
That is a ceiling rather than a forecast, and no mechanism is entitled to it; what it
establishes is that the largest single block of measured deficit sits behind a preprocessing
decision rather than behind the architecture.

## What changes

1. **An inference-time encoder that needs no retraining.** Categorical levels become smoothed
   target statistics computed from the context, which are ordered on the axis the model
   already reads. This works with released checkpoints, so it is measurable immediately and
   its result decides whether the deeper change below is warranted.
2. **Out-of-fold statistics for context rows**, because the naive form is actively harmful
   rather than merely imperfect — see the design note in `inference/categorical.py`.
3. **Only then, a learned categorical path** in the architecture and the prior, if the
   encoder's measured gain leaves enough on the table to justify retraining.

## What this explicitly does not claim

That closing the categorical gap makes fintfm competitive. The eight numeric-only datasets
carry a residual 0.0320 deficit that this proposal does not touch, and that residual is the
honest size of the modelling problem `factorized-attention` (44.x) and §97 are chasing.

## Blocked by

- **Nothing for 47.1-47.3.** The encoder is inference-time and runs against released
  checkpoints, which is the reason it is sequenced first: it produces a measurement before
  any training cost is committed.
- **47.4 is blocked by 47.3's number**, deliberately. Deciding to change the architecture and
  the prior before knowing what the cheap fix recovers would be choosing the expensive
  branch on an assumption, which is the failure mode `docs/results/POSTMORTEM.md` already records
  twice.

## Non-goals

- **One-hot encoding.** It multiplies the column count by cardinality and collides with the
  architecture's `max_features` cap; `Amazon_employee_access` alone would need 7,518 columns
  for one feature.
- **Per-class target statistics for multiclass.** They multiply the feature count by the
  class count for the same reason. Multiclass and regression targets take frequency encoding
  until a measurement asks for more.
- **Tuning the smoothing weight against TabArena.** The benchmark this change exists to
  improve is not a validation set, and fitting a hyperparameter to it would make the
  resulting number meaningless. 10 pseudo-observations is the conventional default and stays
  until a held-out measurement justifies moving it.
- **Closing the accuracy gap.** See the final section of Why: 18% of the deficit sits on
  numeric-only data and this proposal does not address it.
