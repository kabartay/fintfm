# Claims ledger

Every candidate claim for a paper, its evidence, and its status. **Newest evidence wins.**
Status vocabulary:

- **SURVIVES** — measured, replicated across seeds, and not superseded by published work.
- **SINGLE DRAW** — measured once. Not quotable externally until replicated.
- **SUPERSEDED** — someone published it first, or a better method exists. May still be true.
- **RETRACTED** — we asserted it and it is false.
- **OPEN** — proposed, not measured.

Statuses last reviewed 2026-09-09.

---

## Claim 1 — PD term structures produced by per-horizon models are incoherent, and a hazard head fixes it by construction

**Status: SURVIVES.** Evidence: §11 (39% of firms receive a curve where a longer horizon
carries lower cumulative default probability, while the portfolio aggregate stays monotone and
hides it), §20 (the fix), §21 (coherence is free — joint prediction matches per-horizon on
AUC), §26 and §32 (0.00% violations out of time on real data against 39.06%, with only 1.4% of
firms fully monotone under the baseline).

**This is the strongest claim the project has.** It is structural rather than statistical: the
survival parameterisation makes a violation impossible, so no amount of training can produce
one and no monotonicity penalty is needed.

**What weakens it as a *contribution*:** discrete-time hazard models are decades old, and a
competent engineer adds this head in an afternoon. The contribution is not the mathematics — it
is measuring that the field's standard construction is incoherent 39% of the time on real
data, and that IFRS 9 lifetime ECL consumes exactly the curve being broken. Frame it as a
measurement of a live defect, never as a new method.

---

## Claim 2 — A synthetic-only prior transfers to real corporate default data, and cannot have memorised the benchmark

**Status: SURVIVES**, with the transfer weaker than the provenance argument.

Evidence: §14 (domain prior beats a generic one by +0.049 AUC, 3 of 6 cells significant), §15
(pretraining buys calibration: 7× Brier, up to 130× ECE against an untrained control of the
same architecture — but that control still *ranks* at 0.726, so pretraining buys calibration
more than ranking), §18/§19 (the prior matches real task difficulty: logistic-regression AUC
0.743 synthetic against 0.769 real), §30 (a prior widened to reach 0.195% default rates is
worth +0.023 AUC and 2.7× calibration), decision D2.

**The auditable half is the stronger half.** A model that never saw real data cannot have
memorised a benchmark, and the leakage literature quantifies contamination at up to 32 points
of MAPE (§1). That is a property a model-risk reviewer can verify from the training code,
which is unusual and is the part worth writing about.

---

## Claim 3 — Feature conditioning matters more than it should, because financial ratios are pathologically heavy-tailed

**Status: SURVIVES.** Evidence: §35. Three seeds, three panels.

110 of 136 features on V4FinBench have a standard deviation more than **ten times** their
interquartile range, median ratio **240**. A ratio is a quotient and a firm heading for default
is where denominators go small, so this is the normal case rather than the tail case. The
standard PFN treatment — z-score by context mean and standard deviation, clip to ±10 — is
therefore close to useless here: one extreme firm collapses every other firm toward zero, and
the clip bounds the outlier without undoing the collapse.

Rank-transforming first is worth **+0.086 mean AUC** to uniform context and +0.023 to
retrieval out of time, and improves AUC in seven of eight configurations across two further
panels.

**This is the most transferable finding in the project** and the one most likely to be useful
to other people, precisely because it is unglamorous: it says a preprocessing default
inherited from general tabular work is wrong for financial ratios, and quantifies it.

---

## Claim 4 — Context construction dominates architecture choice, and balancing harms it

**Status: SUPERSEDED** as a contribution; the measurement stands.

Evidence: §5, §29 (balanced costs 10-12 AUC points on the survival path; 12 in-context
defaults outrank 1,122), §33 (replicated on three seeds), §35 (on the binary path the
strategies are nearly tied, because those panels lack the positives for "balanced" to reach
50/50 — the effect scales with how extreme the rebalancing *is*, not with the strategy's name).

**Superseded by Kostrzewa et al. (arXiv:2605.10896, May 2026)**, whose prototype undersampling
clusters the majority class and keeps the real row nearest each centroid, with the stated
conclusion that "preserving majority-class structure matters beyond simply increasing minority
exposure" (§36). That is this claim's mechanism, published first.

**Do not present this as a finding of ours.** Cite them for the mechanism and, if anything,
contribute the *narrower* result: that the harm scales with the rebalancing ratio rather than
the strategy, which their ablation does not isolate.

---

## Claim 5 — Query-conditioned retrieval beats global context construction

**Status: RETRACTED.** Settled 2026-09-09 in §38, against us.

| arm | mean AUC | mean ECE | seconds | blind? |
| --- | --- | --- | --- | --- |
| retrieval, 256 groups | 0.8130 ± 0.0069 | 0.0119 | 221 | no |
| **prototype (published)** | **0.8143 ± 0.0038** | **0.0072** | **56** | **yes** |

Prototype ties on mean AUC, **wins significantly at horizon 0** (−0.0077, [−0.0136, −0.0020],
Holm-adjusted p = 0.036), is 1.6× better calibrated, 4× cheaper, more stable across seeds, and
keeps batch independence — where retrieval was measured moving one firm's cumulative PD by
**0.64** depending on its scoring batch (§37).

**What remains true:** retrieval beats *blind uniform* sampling by +0.066 to +0.095 AUC,
replicated across three seeds (§32, §33). That is a real measurement and a useless claim,
because uniform is not the state of the art — Kostrzewa et al.'s prototype context is, and it
is better than ours.

**Do not write this claim in any form.** The residue worth keeping is a negative result: on a
low-default portfolio, conditioning the context on the query does **not** improve on selecting
a structurally representative context once, and it costs batch independence to find out. That
is worth one paragraph in §5.4 of the outline, not a contribution.

## Claim 6 — The model is competitive on accuracy

**Status: RETRACTED, repeatedly, and currently false.**

On the V4FinBench out-of-time split, per-horizon logistic regression leads on mean AUC
**0.8616 to 0.8209** and on calibration by roughly fivefold. Three of four horizon differences
were significant against us at the previous best (§32).

History worth keeping, because it is the shape of the error: §12 claimed an 11.7× calibration
advantage; §16 reduced it to ~2× against a *calibrated* gradient booster; §17 showed a
feature-free constant predictor beats every model on ECE; §25 found every prior comparison had
used sklearn's weakest booster; §27 narrowed the win to portfolios under ~200 obligors. The gap
closed from 0.142 to 0.041 over one day of inference-time fixes, which is real progress and is
not parity.

**A paper must state this in its own abstract.** The accuracy gap is the first thing a reviewer
will compute.

---

## Claim 7 — Low-default portfolios are an underserved segment, and the field's own benchmark did not test them

**Status: SURVIVES.** Evidence: §9 (the authoritative credit-scoring benchmark names the
low-default case as promising and does not test it; their datasets average a 22% default
rate), §4/§8 (firm-level financial data is licence-locked, which is why the space is empty
while energy is crowded), §22 and §36 (V4FinBench's positive rates run 0.19-0.36%, corroborated
against their Table 1).

This is a positioning claim rather than a result, and it is the honest motivation section.

---

## Claim 8 — Conformal PD certificates

**Status: OPEN.** Not started. `openspec/changes/conformal-pd-certificate`.

The intended product (decision D3) and the only claim on this list that would be a *method*
contribution rather than a measurement. **Never claim novelty on the conformal mathematics** —
it is published prior art; the contribution would be the protocol, the pre-registration and
the artefact a validation committee accepts.
