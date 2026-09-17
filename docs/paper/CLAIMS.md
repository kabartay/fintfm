# Claims ledger

Every candidate claim for a paper, its evidence, and its status. **Newest evidence wins.**
Status vocabulary:

- **SURVIVES** — measured, replicated across seeds, and not superseded by published work.
- **SINGLE DRAW** — measured once. Not quotable externally until replicated.
- **SUPERSEDED** — someone published it first, or a better method exists. May still be true.
- **RETRACTED** — we asserted it and it is false.
- **OPEN** — proposed, not measured.

Statuses last reviewed 2026-09-14.

**What changed since 2026-09-09, in one paragraph.** Claim 2 was retracted pending
re-measurement because the pre-fix model was not doing in-context learning at all. It has
now been re-measured in full: the mechanism was found and fixed (§54-§56), a second,
independent instrument (an exactly-known Bayes-optimal-AUC probe) found a severe residual
capacity cap the content-side fix did not touch (§74), seven prior-content hypotheses for that
cap were eliminated one at a time (§58-§77), and an architecture change — two-way cell
attention — closed it in one experiment (§78). Read Claim 2 below before anything else; it is
the whole story in miniature and the reason **every synthetic result in this ledger dated
before 2026-09-14 was measured through a ceiling that no longer exists**, while **no real-data
result has yet been re-measured on the architecture that removed it** (task 39.5, open).

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

## Claim 2 — A synthetic-only prior transfers to real corporate default data

**Status: SINGLE DRAW, and narrower than originally hoped.** The mechanism that broke the
original measurement is fixed and independently confirmed; what transfer exists is now
measured, and it does not generalise the way the original claim assumed.

**The two-part history, in order.**

*Part A — the model could not do in-context learning at all (§47, RETRACTED at the time).*
Shuffling context labels left predictions unchanged (rank correlation 0.977) because every
driver's sign was fixed across every task, so a global rule could be memorised. Fixed by
randomising the sign per task (§47/§48).

*Part B — fixing the sign bug did not fix the architecture (§54, discovered 2026-09-11).* The
row encoder was a **provable symmetric function** of each row's values — `cell_embed` shared
across columns, no positional encoding, pooling over the feature axis — so it could not
represent "column j matters" *by construction*, independent of what the prior taught. Verified
by a closed-form probe: a rule as simple as `x0 - x1` has a symmetric ceiling of exactly 0.5,
and the pre-fix model scored 0.5097. Random per-task column identities (D12, §54/§56) fixed
this for a trivial prior (0.713 to 0.997 AUC) but traded away exact column-order invariance
for a distributional one.

**Then a second, independent instrument found a residual defect the first fix did not touch
(§74, 2026-09-14).** A task with an *exactly known* Bayes-optimal AUC (closed-form Gaussian
mean-shift, verified numerically) showed training on the financial prior caps achieved AUC at
~0.73 **regardless of true task difficulty** — 0.728 achieved at Bayes AUC 0.999, on a probe
with no column-identity structure to solve at all. Seven content-side candidates were
eliminated one at a time using checkpoints that already existed, at zero additional GPU cost
(§58, §62, §64, §65, §72, §76, §77): base rate, raw signal-to-noise, feature cleanliness,
accounting-identity structure (breaking it made things *worse*), and the financial label's
functional form (which does explain *part* of it — financial tasks essentially never reach
realized difficulty above 0.99 even sharpened, §77 — but not all of it).

**Two-way cell attention closed the gap in one experiment (§78).** A checkpoint trained
*exclusively* on the financial prior — the same prior that produced the 0.73 cap — now tracks
the Bayes-optimal curve to within 0.001-0.005 regret across the whole range, confirmed
independently by the antisymmetric probe (0.5352 to 0.9890, closing to the SCM baseline's
0.9932). This settles that the cap was **architectural**, not a property of the prior's
content — every content-side hypothesis this project spent days testing was chasing an effect
whose actual cause was that the encoder had no way to develop column *semantics* from context,
only column *identity*.

**What real transfer exists, measured on the pre-cell-attention architecture (§60-§77):**

- On V4FinBench's published protocol, five folds, tuned baselines, paired bootstrap:
  **fintfm ties logistic regression** (dAP +0.018, Holm p = 0.204) and **loses to all three
  tuned boosters** (dAP -0.14 to -0.17, Holm p < 0.001 each) — §69. A configuration fix
  (dropping retrieval, which actively hurts here; adding ensembling) recovered +0.031 AP,
  significant, without closing the booster gap — §70/§71.
- **The financial prior's content genuinely helps, but only on V4FinBench specifically.** A
  controlled two-factor design (§61/§63) found domain content worth +0.098 AP at matched base
  rate (Holm p = 0.002) and density worth nothing (+0.005, p = 0.895). But the same comparison
  on two *other* real credit panels (Polish, Taiwan bankruptcy, §73/§75) found the **opposite
  ordering** — pure generic-SCM beat the financial-content checkpoint on all four measured
  cells, by up to +0.065 AP on Taiwan. The financial generator's benefit does not travel past
  the one benchmark's own feature-engineering conventions.
- **Calibration generalises where discrimination does not.** fintfm has the best or near-best
  ECE on all four real panels measured (§73), independent of which prior wins on
  discrimination there. This is the one piece of the original claim that has replicated
  cleanly across every dataset tried.

**What is not yet measured**: any of the above, on the architecture that removed the capacity
cap. `cell-attention-and-task-inference` task 39.5 is open. **Do not write any version of this
claim for external use until it is** — this project has already been burned twice by writing
a transfer claim from a checkpoint that turned out not to support it (§47, §74), and the
correct number of times to be burned by the same mistake is zero.

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

**Regime-dependence found 2026-09-13/14 (§70), sharpening the negative result further.** On
V4FinBench's published protocol at its real base rate (0.380%), retrieval alone drops AP by
**0.133** relative to uniform sampling — actively harmful, not merely tied. Not rescued by
wider context or ensembling; every combination including retrieval scored below the uniform
baseline. The mechanism is unresolved (a centroid-averaging hypothesis was proposed but the
verification run was killed mid-flight during a machine-load safety stop and never completed
— stated here so it is not mistaken for a ruled-out hypothesis). **Retrieval's sign now
depends on the regime it is measured in** (helps at whatever base rate §32/§33 measured,
actively hurts at V4FinBench's 0.38%), which is itself worth a sentence in any paper that
cites this project's earlier retrieval numbers.

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

**A second, independently-produced measurement of the same gap (§60/§69/§71, 2026-09-13/14),
on a different harness — V4FinBench's own published protocol rather than this project's
out-of-time split, five folds, tuned baselines, average precision rather than ROC-AUC (D13:
AUC's chance floor is 0.5 regardless of prevalence, so at V4FinBench's 0.38% base rate it
compresses the whole usable range into its top few percent).** fintfm statistically ties
logistic regression and loses to CatBoost/XGBoost/LightGBM by 0.14-0.17 AP, all significant at
Holm p < 0.001. **The first version of this same comparison, on ROC-AUC with untuned
baselines, misranked fintfm second of five** where AP with tuned baselines puts it third,
behind both boosters that ROC-AUC's untuned reading had understated — a concrete demonstration
of why AP-first reporting at low prevalence is not a stylistic preference.

Both measurements point the same direction from different angles and should both be cited if
either is: the out-of-time split (this entry, §32) for the temporal-holdout argument, the
published-protocol number (§60/§69) for direct comparability to V4FinBench's own paper.
**Both have now been re-measured on the cell-attention architecture (§80, 2026-09-15), and
the deficit survives.** On the published protocol at five folds with tuned baselines, the new
architecture loses to LightGBM/CatBoost/XGBoost by **0.204/0.234/0.236 AP**, every one Holm
p < 0.001 — a gap of the same character and larger magnitude than the 0.14-0.17 recorded above,
because both arms were scored at the reduced context §79 forces. The one status change is
against logistic regression: §69's **tie** (Holm p = 0.204) becomes a significant **fintfm win**
of +0.0307 AP. So Claim 6 remains RETRACTED and false as stated — beating logistic regression
is not "competitive on accuracy" when every tuned booster leads by more than the entire
fintfm-to-LR margin, roughly sevenfold.

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

---

## Claim 9 — Calibration generalises across independent real credit panels; discrimination does not

**Status: SURVIVES, on the pre-cell-attention architecture.** Evidence: §73, four real panels
(V4FinBench, Polish bankruptcy at three horizons, Taiwan bankruptcy), against tuned classical,
random forest and gradient-boosting baselines.

fintfm has the **best or near-best ECE on all four panels**, independent of which prior wins on
discrimination there (§75 found the *opposite* prior wins discrimination on three of the four).
This is the first time the calibration thesis (§12) has been tested outside V4FinBench, and it
held without exception, while the discrimination gap to boosters varies **1.7x to 7.2x** across
the same four panels at similar base rates — not explained by class balance, so it tracks
dataset structure specifically.

**This is arguably the strongest surviving empirical claim in the project besides Claim 1.**
It is also the one whose mechanism is least understood: why calibration transfers when
discrimination does not is an open question this ledger does not yet have an answer to, and it
connects directly to task 39.7's proposed task-representation probe (does the model form a
task posterior, separate from how well that posterior discriminates?).

**Caveat**: single 70/30 split per dataset, not the five-fold paired-bootstrap standard the
V4FinBench numbers meet (§69). Directional, not settled at this precision.

---

## Claim 10 — Two-way cell attention is a real, reproducible architectural fix for a severe capacity defect

**Status: SURVIVES on synthetic AND real data, at matched context. Evidence: §78 (synthetic), §80 (real, five folds).** Evidence: §78, one
experiment (`n_cell_blocks=1`, `cell_labels=True`, both `p_financial` arms), confirmed by two
independent instruments (a Bayes-ceiling probe with a closed-form, numerically-verified target,
and the pre-existing antisymmetric probe).

A checkpoint trained exclusively on the financial prior — the same prior that produced a
severe, well-characterised capacity cap (§74: regret 0.234-0.277 at Bayes AUC 0.90-0.999,
independent of seven content-side interventions tried against it, §58-§77) — closes to regret
0.001-0.005 across the same range after adding row-attention-within-feature blocks and
per-cell label injection before pooling. The SCM arm, already near-optimal, shows no
regression.

**Why this belongs in a methods section rather than only a results table**: it settles, with
a controlled before/after on the identical prior, that the architecture — not the prior's
content — was the binding constraint. That is a stronger and more general claim than "our
prior transfers," because it says something about *what an in-context tabular model needs
architecturally* to learn column semantics from context, independent of which prior it is
trained on.

**What would strengthen this before it is quotable**: replication at the exact protocol §74
used (the reported run deviated to `--batch-size 4 --n-rows-choices 256,512` after a CUDA OOM,
itself caused by an unrelated bug — a hardcoded evaluation batch size independent of training's
own, now fixed); a second seed; and a second seed. One honest anomaly is on the
record rather than smoothed over: `symmetric_count` is slightly lower on both new checkpoints
than their old-architecture counterparts, unexplained.

**Task 39.5 closed, 2026-09-15 (§80): the fix transfers.** Two complete five-fold V4FinBench
runs on the full 1.0M-row panel, differing only in `n_cell_blocks`/`cell_labels`, give cell
attention **+0.0486 mean AP, 5 folds of 5, every 95% CI excluding zero, every Holm-corrected
p < 0.001.** The synthetic result did not dissociate on real data — the specific failure §47
and §74 both exhibited and which this claim was explicitly held open against.

**Attribution resolved, 2026-09-17 (§91): the two changes are jointly necessary.** A matched
pair at the full protocol, differing only in `--cell-labels`, gives cells+labels **+0.514 mean
AUC** over cells-alone on the Bayes-ceiling probe — because the unlabelled variant scores
**below chance** (0.4407-0.4575) at every difficulty, producing inverted rankings. So neither
half of this claim's architecture change can be dropped: row-attention-within-feature without
a per-cell label is worse than the architecture it replaced. The mechanism is unexplained.

**Status upgrade, 2026-09-16 (§84): the architecture wins best-vs-best on real data.** Each
architecture scored at *its own best measured context* on identical full-panel rows gives cell
attention **+0.0417 AP, 5 folds of 5**. The architecture effect (+0.039 to +0.049 at matched
context) is 6-7x the entire context effect (0.0069 for the old architecture, -0.0032 for the
new). Nothing about this claim now rests on a handicapped comparison.

**A caveat this claim used to carry, now retracted (§82).** It previously said the new
architecture's best reachable score was below the old architecture's best recorded score
(0.1941 against §71's 0.2116). That was wrong twice over: 0.2116 is §71's *fold 0*, not its
five-fold mean of 0.1676, and §69-§71 ran on a **10x subsample** (105,900 test rows against
§80's 1,000,087), so the numbers were never comparable. The full-panel measurement of the
context effect is +0.0069 from 1000 to 2000, not the -0.066 inferred. That re-scoring is now done (§84) and the
question is answered: the architecture improves this project's real-data standing, by +0.0417
AP best-vs-best on 5/5 folds. What remains open is narrower and is an explanation rather than
a measurement — cell attention prefers *less* context than the old architecture, which
contradicts the mechanism §83 predicted and is unexplained (task 39.26).

The engineering blocker that produced the original caveat is also gone (§81): chunking
row-within-feature attention over `F` is identity-preserving, 6.2x smaller and 1.6-2.4x
faster, and `max_context=4000` runs. §83 then measured the old architecture there and found it
slightly *worse* than 2,000, so the configuration §80 mourned losing turned out not to be
worth having.
