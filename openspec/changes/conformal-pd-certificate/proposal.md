# The certificate: conformal PD intervals with honest refusal

## Why

This is the product (`docs/design/DECISIONS.md` D3). Everything upstream exists to make it cheap to
produce. A credit risk function under supervisory obligation does not buy a point estimate;
it buys the ability to answer "where can I trust this, and where must I not" in a document a
validation committee accepts without the vendor in the room.

The machinery is already proven in a sibling project: `finkele-axiom` is a validation
protocol producing a certificate with conformal coverage, out-of-distribution escalation and
honest refusals, generalised across physics. Pointing it at credit risk is a port, not
research — which is why this is the highest-leverage work in the repository once Phase 1
returns a positive result.

**Never claim novelty on the conformal mathematics.** It is published prior art. The
contribution is the protocol, the pre-registration, the certificate and the packaging.

## What

- Split-conformal PD intervals with a stated coverage guarantee, calibrated on a
  pre-registered slice.
- Coverage measured under **regime shift**, not just in-distribution: calibrate on one
  economic period, test on another.
- Out-of-distribution detection over the feature space, with escalation — a firm unlike
  anything in the context is **refused**, not scored. A certificate that cannot say "no" is
  not evidence.
- A generated `certificate.md` in which every clause is a sentence about a named field in a
  `results.json`, so a third party can re-derive it.

## Non-goals

- Not a UI. The deliverable is files.
- Not LGD or EAD; PD only, initially.
- Not importing `finkele-axiom` as a dependency — it is AGPLv3 and this repo is Apache-2.0.
  **An import would propagate AGPL.** Reimplement, or exchange files, exactly as that repo's
  own licence boundary requires.

## Falsified by

If conformal intervals on this model are so wide as to be operationally useless — an interval
spanning 0% to 40% PD tells a lender nothing — then the honest certificate says the model is
not fit for purpose, and that is a real result about the model rather than a failure of the
protocol.

## Blocked by / blocks

- **Blocked by** `phase1-prior-ablation` and `fitted-calibration`.
- **Blocked by** `time-based-evaluation` for the regime-shift half, which needs period labels.
- **Blocks** any commercial conversation, because it is the thing being sold.
