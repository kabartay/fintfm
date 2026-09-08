# Establish why the calibration advantage exists, and whether it is ours

## Why

`docs/FINDINGS.md` §12 measured a 2.3-11.7× calibration advantage over gradient boosting and
§13 traced it to Bayesian shrinkage: a prior-fitted network approximates the posterior
predictive, which is calibrated by construction and prior-dominated at small *n*. The
measured decay of the advantage with *n* matches that prediction.

Two questions decide how much this is worth, and **neither is answered**:

1. **Does it survive a calibrated incumbent?** Platt or isotonic scaling on gradient boosting
   is cheap and standard. The argument that it fails in our regime — post-hoc calibration
   needs held-out events, and a low-default portfolio has none — is compelling and untested.
2. **Is it ours, or generic to PFNs?** If the generic-prior variant calibrates equally well,
   then TabPFN and TabFM share the property and our differentiator is the domain prior plus
   the certificate, not calibration. That is a materially weaker position and it must be
   known before any pitch rests on it.

Answering these is worth more than any accuracy improvement, because it determines whether
the measured advantage is a moat or a footnote.

## What

- Add a **calibrated gradient boosting arm** (Platt and isotonic, fitted on a validation
  split carved from the training set) to the sample-efficiency probe, so the comparison is
  against the best available incumbent rather than a straw one.
- Report the **discrimination/calibration frontier** rather than either metric alone, since
  §13 shows the honest picture is Pareto dominance at n≈100 and a trade-off above it.
- Compare **financial versus generic prior on calibration** at matched compute, to separate
  method from domain.
- Measure **prediction spread** alongside calibration everywhere, so conservatism cannot be
  reported as if it were skill.

## Non-goals

- Not improving calibration further. This change is about establishing what is true.
- Not conformal intervals; that is `conformal-pd-certificate`, which should sit on top of a
  mechanism that is understood.

## Falsified by

- If calibrated gradient boosting closes the gap at n≈100, the advantage is a comparison
  artefact and §12 must be rewritten.
- If the generic prior calibrates as well as the financial one, calibration is not a
  domain-specific advantage. Say so in `docs/STRATEGY.md` and move the differentiator to the
  certificate alone.

## Blocked by / blocks

- **Blocked by** `phase1-prior-ablation` finishing, which supplies the generic-prior
  checkpoint for question 2.
- **Blocks** any external claim about calibration, and `conformal-pd-certificate`, which
  should not be built on an unexplained empirical advantage.
