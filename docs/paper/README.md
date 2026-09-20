# Paper workspace

Material for a potential paper on FinTFM. **Not a draft yet, and deliberately not one** — this
folder exists so the decision of *what could honestly be claimed* is made from the evidence
ledger rather than from memory at writing time.

| file | what it is |
| --- | --- |
| [CLAIMS.md](CLAIMS.md) | every candidate claim, its evidence, and its status. The load-bearing file. |
| [OUTLINE.md](OUTLINE.md) | the structure a paper would take, and which claim carries each section |
| [RELATED_WORK.md](RELATED_WORK.md) | who has done what, and what is left for us |
| [LIMITATIONS.md](LIMITATIONS.md) | the section that has to be written first, not last |
| [FIGURES.md](FIGURES.md) | figures and tables a paper needs, and which command produces each |

## Current publishability, as of 2026-09-20

**Yes — for one specific paper, and no for the one this project set out to write.** The
evidence now settles a framing question [OUTLINE.md](OUTLINE.md) had left open since §78.

| candidate paper | state |
| --- | --- |
| **The architecture diagnosis** (symmetric-encoder limit, Bayes-ceiling probe, cell-attention fix) | **writable now.** Provable ceiling, closed-form instrument, seven eliminated alternatives, controlled fix, real-data confirmation at +0.042 AP on 5/5 folds |
| **A synthetic prior transfers to credit risk** | **not writable.** §96 found the deficit is general rather than credit-specific; §98 placed the model 93rd of 95 on the field's own benchmark. The claim is comparative and the comparison is lost |
| **Term-structure coherence** (Claim 1) | **writable now, separately.** Small, structural, independent of the above. Folding it into the diagnosis paper would dilute both |

**What changed the answer was external measurement.** §98 is the first placement in this
project's history whose baselines and protocol belong to someone else, and §100 is the first
decomposition of a deficit into a part that is the model's and a part that is not — 82% of the
TabArena gap is categorical preprocessing, leaving a **−0.0320** numeric-only residual that is
2.8× smaller than the headline. A diagnosis paper needs a characterised residual, and now has
one.

## The rule for this folder

> **No claim enters without a `docs/FINDINGS.md` section number, and no number enters without
> the command that produced it.**

This is not ceremony. Over 2026-09-08/09 this project produced five wrong diagnoses in one day
(`docs/POSTMORTEM.md`), every one caught by measurement and none by review. A paper drafted
from recollection would have stated at least three claims that were true when believed and
false a day later — including one that a retrain was spent on.

A second rule follows from §36: **read the source, not a summary of it.** A fetched summary of
the V4FinBench paper invented a results table, invented default rates tenfold too high, and
inverted whether their TabPFN was fine-tuned. Anything cited here is cited from the pages.

## Previous assessment, as of 2026-09-10

**§47 changed the answer from "not yet" to "not close".** The model was not doing in-context
learning: shuffling its context labels left predictions unchanged and AUC higher than with
true labels. Every accuracy claim below was measured on that model. The cause is found and
fixed — the prior's feature-to-label direction was constant across tasks — and verification is
in flight, but nothing accuracy-related is publishable until a checkpoint demonstrably reads
its context.

The coherence claim (Claim 1) is unaffected: it is a property of the output parameterisation
and holds whatever the model has learned.

## Previous assessment, as of 2026-09-09

**Not yet, and the reason is specific rather than modest.**

| dimension | us (best measured) | the incumbent | verdict |
| --- | --- | --- | --- |
| out-of-time mean AUC | 0.8143 | 0.8616 (per-horizon logistic regression) | **behind by 0.047** |
| out-of-time calibration (ECE) | 0.0072 | ~0.0018 | **behind by ~4×** |
| coherence violations | **0.00%** | 39.06% | **ahead, by construction** |

Two further blockers, both concrete:

1. **No comparable number exists yet.** Our protocol is out-of-time; V4FinBench's published
   protocol is 5-fold company-grouped cross-validation, its horizon tasks are built on
   different rows, its context is 10,000 rows, and its TabPFN is fine-tuned on the data (§36).
   Nothing we have can be placed against their table until their protocol is reproduced —
   `openspec/changes/public-benchmark-claim` task 33.2. They release fold indices.
2. **The context-construction insight is not ours, and neither is the better method.** §29's
   mechanism was published in May 2026 as prototype undersampling (§36), and when implemented
   from their description it **beat our query-conditioned retrieval** on accuracy at the first
   horizon, on calibration, on cost and on batch independence (§38). Our best out-of-time
   configuration now *uses their context construction*. Claim 5 is retracted.

What is publishable *today* is the coherence result and the synthetic-only provenance
argument. Both are real; neither is an accuracy claim; and a paper that leads with accuracy
would be overclaiming by 0.041 AUC.
