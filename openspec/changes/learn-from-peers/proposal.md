# Take the ideas the peer models have already validated, and recalibrate what scale can buy

## Why

§107 removed the specialist reading of this project's TabArena placement: median rank 94 of
95, credit panels at 83–95, and the apparent "top-21 on its best datasets" traced to one
dataset, one fold, and a 0.0026 AUC margin. There is no peak to protect, so the uniform
~0.035 ROC-AUC residual (§101) is the whole target and breadth is the only supported position.

§104 closed `column_id_dim`, the last untuned lever that had moved real-data accuracy. §102
found no axis in the residual and §103 weakened the mechanism it proposed. That left **scale**
as the sole standing hypothesis — and §108's first attempt at it went the wrong way, on a
confounded design.

Seven peer models were opened on 2026-09-21 (`docs/REFERENCES.md`). One of them, **Nori**, is
close enough to this project's design to be read as a controlled experiment someone else has
already paid for: synthetic-only pretraining, in-context prediction, alternating
feature/sample attention, and a distributional head. It publishes a scaling curve.

**That curve is the reason this proposal exists.**

| Nori variant | parameters | TabArena R² | overall R² (95 tasks) |
| --- | --- | --- | --- |
| nori-6m | 6M | 0.8069 | 0.7567 |
| nori-30m | 30M | 0.8099 | 0.7588 |
| nori-100m | 100M | 0.8118 | 0.7601 |

**16.7× parameters buys +0.0049 R² on TabArena and +0.0034 overall.** The metric is not ours
(R² on regression against ROC-AUC on binary), so the number does not transfer. The *order of
magnitude* does: in this model class, a well-executed order-of-magnitude scale-up returns
single-digit thousandths. Our residual is 0.035.

**A perfectly executed scaling programme would not have closed the gap.** That was not knowable
before reading their curve, and it changes the roadmap's priority order rather than merely
adding to it.

## What this proposal is not

It is not permission to ingest anything. `CLAUDE.md` forbids code, weights and training data
from any tabular-foundation-model product entering this repository, and that prohibition is
independent of licence. Nori's code **and** weights are Apache-2.0 — checked separately, as
the rule requires, on 2026-09-21 — which makes *evaluating* it permissible and changes nothing
about ingestion. Every task below is an independent implementation of a published idea.

It is also not a claim that these ideas will work here. Each is a hypothesis with a named
falsification, and several may fail for reasons specific to a 136-feature financial prior.

## Non-goals

- **Not adopting any third-party implementation.** `CLAUDE.md`'s boundary stands and is
  independent of licence: no code, weights or training data from Nori, ConTextTab / SAP-RPT-1,
  TabSTAR, TabFlex, Orion-MSP, iLTM, TabPFN, TabICL or TabDPT may enter this repository. Every
  idea here is reimplemented from a public description. Nori's code and weights were both
  checked and are Apache-2.0 — that makes *evaluating* it permissible and changes nothing.
- **Not a claim that any of these ideas will work here.** Each task names its own
  falsification, and a null is a result. A 136-feature financial prior is not the setting any
  of these were tuned for.
- **Not abandoning scale.** 48.2 recalibrates the *expected return*; it does not argue for zero
  parameters. The matched-task medium run (§108's correction) still has to land, because a
  confounded negative is not a negative.
- **Not a pivot to real-table pretraining.** 48.7 records the counter-thesis and prices it;
  it does not adopt it. That would trade this project's only durable differentiator for
  accuracy it would still not win on.

## Blocked by

- **48.1 is blocked by nothing** and is the first thing to run — it is cheaper than §108's
  failed arm (2.6M against 4.98M parameters) and tests the axis §108 did not vary.
- **48.2 blocks nothing but should precede any further scale spend**, since it decides whether
  `docs/STRATEGY.md`'s 10–50M target survives.
- **48.3 is blocked by 46.7**: the binned head's real-data numbers do not exist yet, so a
  quantile head would be replacing something unmeasured.
- **48.4/48.5/48.6 belong to `mechanism-diverse-prior`** and should be sequenced against it
  rather than run in parallel, or two prior changes will confound each other.
