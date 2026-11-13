# Claims ledger

Every candidate claim for a paper, its evidence, and its status. **Newest evidence wins.**
Status vocabulary:

- **SURVIVES** — measured, replicated across seeds, and not superseded by published work.
- **SINGLE DRAW** — measured once. Not quotable externally until replicated.
- **SUPERSEDED** — someone published it first, or a better method exists. May still be true.
- **RETRACTED** — we asserted it and it is false.
- **OPEN** — proposed, not measured.

Statuses last reviewed 2026-11-13 against `docs/results/FINDINGS.md` §1-§224 (Claim 2 note added
2026-11-05). Numbers and section references were also re-checked on 2026-10-15; corrections made
then carry that date. Claims 16-18 (§157-§164) were added 2026-10-16 and Claim 19 (§204-§224)
on 2026-11-13. Notes dated 2026-11-13 carry the section that supports them; the original text
stays.

> **Every claim in this ledger is a claim about FinTFM, the synthetic-only family.** Decision
> **D17** (2026-10-06) opened a second family, **FinTFM-R**, trained on real data (SEC EDGAR;
> Freddie Mac once licensed). No FinTFM-R checkpoint exists yet, so nothing below has been
> measured on one. When the first does: **the provenance half of Claim 2 does not transfer to
> it, by construction** — "the model cannot have memorised a benchmark because it never saw
> real data" is false of FinTFM-R and must never be restated for it. Any FinTFM-R evidence
> added here carries the family name in the evidence line, and any benchmark it is scored on
> must first be checked for SEC-filer overlap (`CLAUDE.md`, "Two families").
>
> (Updated 2026-10-16: FinTFM-R checkpoints now exist and are scored in Claim 17, §157, §160,
> §161. Separately, **opt-in dataset-specific fine-tuning** (0.7.0, §163, Claim 16) gives a
> FinTFM checkpoint gradient steps on the caller's training rows. A fine-tuned model has seen
> those rows, so the provenance half of Claim 2 covers the **zero-shot** model only, the default
> (decision D18).)

**What changed since 2026-09-14, in one paragraph.** The architecture fix was measured on
one real panel, V4FinBench (§80/§84, Claim 10), five candidate explanations for the residual accuracy deficit
were measured and eliminated (training volume, context size, marginal augmentation, z-clip,
prior domain), and the project was measured **externally for the first time**: TabArena places
it 93rd of 95 [§225: read from a table numbered from 0; §98's own leaderboard is not stored] (§98, new Claim 11; 94th of 95 on a later board snapshot, §122). §100 then
decomposed that result and found much of the deficit is a *preprocessing* artefact rather than
an architectural one: it correlates −0.668 with log categorical cardinality, and 82% of the
distance to the method ranked #89 sits on datasets with categorical columns. That 82% is a
ceiling; native categorical handling recovered 34% (§101). A multiclass head was trained and verified against its own untrained
control (§99, new Claim 12). Claim 6 is unchanged in status and now has a third, external
measurement behind it.

**What changed since 2026-09-09, in one paragraph.** Claim 2 was retracted pending
re-measurement because the pre-fix model was not doing in-context learning at all. It has
now been re-measured in full: the mechanism was found and fixed (§54-§56), a second,
independent instrument (an exactly-known Bayes-optimal-AUC probe) found a severe residual
capacity cap the content-side fix did not touch (§74), seven prior-content hypotheses for that
cap were eliminated one at a time (§58-§77), and an architecture change — two-way cell
attention — closed it in one experiment (§78). Read Claim 2 below before anything else; it is
the whole story in miniature and the reason **every synthetic result in this ledger dated
before 2026-09-14 was measured through a ceiling that no longer exists**. Task 39.5 — the
real-data re-measurement this paragraph used to flag as open — **closed on 2026-09-15 (§80)
and the fix transferred**: +0.0486 AP at matched context, +0.0417 best-vs-best, 5 folds of 5.

---

## Claim 1 — PD term structures produced by per-horizon models are incoherent, and a hazard head fixes it by construction

**Status: SURVIVES.** (2026-11-13, §228, §229: §11's 39% of firms does not reproduce on the stored checkpoint and is no longer evidence. The out-of-time evidence survives a split with no firm on both sides and a six-year label embargo: per-horizon logistic regression violates 40.3% of horizon steps and leaves 98.7% of firms non-monotone, against 0% for the hazard head. Quote those.) Evidence: §11 (39% of firms receive a curve where a longer horizon
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

**Status: SINGLE DRAW, narrower than originally hoped, and NARROWED AGAIN 2026-09-21.**

**Scope, after D17:** this claim is about **FinTFM** only. It is not a claim about FinTFM-R,
which trains on real data and for which the provenance argument does not exist. Quoting this
claim without the family name is the specific error `CLAUDE.md`'s "Two families" section
exists to prevent.

**The "synthetic-only" half is no longer ours to claim.** Synthefy's Nori is synthetic-only,
in-context, single-forward-pass, **Apache-2.0 in both code and weights** (checked separately),
and further along — alternating feature/sample attention, an 8-edge-function SCM prior, 9
target families, a learnability filter, a 999-quantile head, and a published scaling curve.
The *technique* is public and permissively licensed. What remains defensible is the
**financial** generative story, the provenance *argument* (not the mechanism), and the
calibration result. Any draft must make the claim in that narrower form; see
`docs/paper/RELATED_WORK.md`.

**Update 2026-11-05 (peer read, `docs/paper/RELATED_WORK.md`): narrower again, in two ways.**
Two more peers now ship synthetic-only, in-context models (TabAbax, and Nori 1.5
with a third-party classifier head), so the mechanism is a field default, not a differentiator.
And **a model that described its pretraining data as synthetic, LimiX-2, is the subject of an open
contamination report on TabArena** (autogluon/tabarena issue #648; the team attributes it to
pipeline bugs). The surviving version of the provenance argument is the *auditable* one: FinTFM
publishes its generator and never touches real data, so the label can be checked. This is a
positioning claim, not a measurement, and applies to FinTFM only.

**Everything below still holds as measured.** The mechanism that broke the
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
(§58, §62, §64, §65, §72, §76, §77), among them base rate, raw signal-to-noise, feature cleanliness,
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

- On V4FinBench's published protocol, five folds, tuned baselines, paired bootstrap, on a
  **10x subsample (105,900 test rows, §82)**: at library defaults **fintfm ties logistic
  regression** (dAP −0.0127, Holm p = 0.309) and **loses to all three tuned boosters** (dAP
  −0.17 to −0.21, Holm p < 0.001 each), §69. A configuration fix (dropping retrieval, which
  actively hurts here; adding ensembling) recovered +0.031 AP, significant, without closing the
  booster gap: after it, fintfm still ties logistic regression (dAP +0.018, Holm p = 0.204) and
  trails the boosters by 0.14 to 0.17 AP, Holm p < 0.001 each, §70/§71 on the same subsample.
  (Corrected 2026-10-15: this bullet previously credited §71's post-fix numbers to §69.)
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

**Re-measured on the fixed architecture, and the conclusion got worse, 2026-09-17/20.** This
entry used to say the above was unmeasured on the post-§78 architecture and to withhold the
claim until it was. That measurement is done, and three results now bear on it:

- **§96** — the accuracy deficit is **general, not credit-specific**: the model is behind on
  14 of 15 public OpenML binary tasks, and the prior's domain is worth only +0.010.
- **§98** — externally, on TabArena, **93rd of 95 [§225: read from a table numbered from 0; §98's own leaderboard is not stored]** (94th of 95 on a later board snapshot,
  §122), losing to every linear baseline.
- **§100** — the deficit tracks categorical cardinality at **−0.668**, i.e. much of it is not
  about domain transfer at all.

**Do not write any version of this claim for external use.** The reason has changed and
strengthened: it is no longer "not yet measured" but "measured, and the comparison is lost".
What survives is narrower and still true — the financial prior helps *on V4FinBench
specifically* (§61/§63) and loses to a generic SCM prior on two other credit panels
(§73/§75). A transfer claim scoped to one benchmark's feature conventions is not the claim
this project set out to make, and [OUTLINE.md](OUTLINE.md) now frames the paper around the
architecture diagnosis instead.

**Update 2026-11-13 (§169, §205, §216): the tag stays SINGLE DRAW, and the conclusion is
unchanged.** Nothing measured since 2026-09-20 supports a transfer claim. Four credit datasets
outside V4FinBench have now been scored, with three out-of-time or stratified splits each and one
seed, and FinTFM is below untuned logistic regression on mean AP on all four (ULB 0.6830 against
0.7489, SBA 7(a) 0.1095 against 0.1295, FDIC 0.1688 against 0.1906, Estonia 0.0028 against 0.0056;
§216, from §174 and §177). On V4FinBench the best zero-shot lever, partitioned contexts, scores
0.2362 mean AP against 0.1986 (§169, five folds, no training), which is a gain on the one
benchmark whose conventions the financial prior was built around; untuned CatBoost on the same
folds is 0.4005 (§167). On the prior's own tasks the trained head beats a linear probe (§203),
so §130's opposite result on V4FinBench is a real-data gap, and §209 traces the probe's real-data
advantage to label count. None of these is evidence for the claim as first stated.

This project has already been burned twice by writing a transfer claim from a checkpoint that
turned out not to support it (§47, §74). The correct number of times to be burned by the same
mistake is zero.

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
V4FinBench's published protocol, on §70's single fold of the 10x subsample (20,917 test rows,
79 positives, prevalence 0.380%; §82), retrieval alone drops AP by
**0.133** relative to uniform sampling — actively harmful, not merely tied. Not rescued by
wider context or ensembling; every combination including retrieval scored below the uniform
baseline. The mechanism is unresolved (a centroid-averaging hypothesis was proposed but the
verification run was killed mid-flight during a machine-load safety stop and never completed
— stated here so it is not mistaken for a ruled-out hypothesis). **Retrieval's sign now
depends on the regime it is measured in** (helps at whatever base rate §32/§33 measured,
actively hurts at that fold's 0.38%), which is itself worth a sentence in any paper that
cites this project's earlier retrieval numbers.

## Claim 6 — The model is competitive on accuracy

**Status: RETRACTED, repeatedly, and currently false.**

On the V4FinBench out-of-time split, per-horizon logistic regression leads on mean AUC
**0.8616 to 0.8143** (prototype context, three seeds, §38) and on calibration by roughly
fourfold (ECE ~0.0018 against 0.0072). Three of four horizon differences were significant
against us at the previous best (§32). (Corrected 2026-10-15: this entry previously quoted
0.8209, a single-seed draw that §38 retracted to 0.8130 ± 0.0069, and "fivefold".)

History worth keeping, because it is the shape of the error: §12 claimed an 11.7× calibration
advantage; §16 reduced it to ~2× against a *calibrated* gradient booster; §17 showed a
feature-free constant predictor beats every model on ECE; §25 found every prior comparison had
used sklearn's weakest booster; §27 narrowed the win to portfolios under ~200 obligors. The gap
closed from 0.142 to 0.047 mean AUC (FINDINGS scorecard; §35, §38) over one day of inference-time fixes, which is real progress and is
not parity.

**A paper must state this in its own abstract.** The accuracy gap is the first thing a reviewer
will compute.

**A second, independently-produced measurement of the same gap (§60/§69/§71, 2026-09-13/14),
on a different harness — V4FinBench's own published protocol rather than this project's
out-of-time split, five folds, tuned baselines, on a 10x subsample (105,900 test rows, §82),
average precision rather than ROC-AUC (D13: AUC's chance floor is 0.5 regardless of
prevalence, so at the subsample's 0.38% base rate it compresses the whole usable range into its
top few percent).** After §71's configuration fix, fintfm statistically ties logistic
regression (dAP +0.018, Holm p = 0.204) and loses to CatBoost/XGBoost/LightGBM by 0.14-0.17
AP, all significant at Holm p < 0.001. **The first version of this same comparison, on ROC-AUC with untuned
baselines, misranked fintfm second of five** where AP with tuned baselines puts it third,
behind both boosters that ROC-AUC's untuned reading had understated — a concrete demonstration
of why AP-first reporting at low prevalence is not a stylistic preference.

**A third measurement, external and adversarial (§98/§100, 2026-09-20).** TabArena, 26
datasets scored at §98 (§100's head-to-head counts below are over 27; FINDINGS gives both), the
field's own protocol and leaderboard: mean ROC-AUC **0.7642** against tuned
logistic regression's 0.8169 and default random forest's 0.8241, winning 2 of 27 and 5 of 27
respectively. This is the first accuracy measurement in this ledger whose baselines this
project did not run itself, and it agrees with the other two.

**But §100 changes what the gap is evidence *of*.** Split by categorical content, the deficit
is **−0.0320** on the eight datasets with no categorical columns and **−0.0894** on the nine
that are mostly categorical; against maximum level count it runs −0.0320 / −0.0437 / **−0.1109**
and correlates **−0.668** with log cardinality. The architecture reads every cell as an ordered
scalar, so the label-encoded integration imposed a false order — and that, not the model, is
where most of the measured distance lies. **The honest size of the modelling deficit is the
numeric-only figure, −0.0320, which is 2.8× smaller than the headline.** Any future statement
of this claim must quote the split, not the mean alone.

Both measurements point the same direction from different angles and should both be cited if
either is: the out-of-time split (this entry, §32) for the temporal-holdout argument, the
published-protocol number (§60/§69, a 10x subsample of 105,900 test rows, §82) for direct comparability to V4FinBench's own paper.
**Both have now been re-measured on the cell-attention architecture (§80, 2026-09-15), and
the deficit survives.** On the published protocol at five folds with tuned baselines, the new
architecture loses to LightGBM/CatBoost/XGBoost by **0.204/0.234/0.236 AP**, every one Holm
p < 0.001. The gap is of the same character; its magnitude cannot be compared with the
0.14-0.17 recorded above, because §69-§71 scored a 10x subsample (105,900 test rows) and §80
the full 1,000,087-row panel (§82: tuned CatBoost scores 0.3425 AP on the subsample and 0.4277
on the full panel). (Corrected 2026-10-15: this sentence previously attributed the larger gap
to the reduced context §79 forces.) The one status change is against logistic regression: the
earlier **tie** (§69 at library defaults, Holm p = 0.309; §71 after the configuration fix,
p = 0.204, both on the subsample) becomes a significant **fintfm win** of +0.0307 AP. So Claim 6 remains RETRACTED and false as stated — beating logistic regression
is not "competitive on accuracy" when every tuned booster leads by more than the entire
fintfm-to-LR margin, roughly sevenfold.

**Updated 2026-10-16 (§162, §163): the gap is to the boosted trees, not to logistic
regression.** A fold-0 diagnostic at the 200k-row development cap had untuned logistic
regression ahead, 0.1658 AP against FinTFM's 0.1442. That held only at the cap (34,906 test
rows, a training pool of ~106k rows). On the full 1,000,087-row panel, five folds, the same
checkpoint zero-shot scores 0.1986 mean AP against untuned logistic regression's 0.1614, ahead
on every fold (§163's table). Against untuned boosters on the same five folds (library
defaults): CatBoost at 1,000 iterations 0.4005, LightGBM 0.2666 and XGBoost 0.2229 at 100 trees
each. CatBoost wins every fold. Claim 6 stays RETRACTED. (§167, 2026-10-16: all on V4FinBench's
published features, which keep `operational_status`. LightGBM at 1,000 trees and learning rate
0.05 scores 0.3908 on the same features, so the field is about 0.39-0.40 mean AP; the 100-tree
LightGBM and XGBoost figures are under-trained defaults and must not be quoted as the field.)

**Updated 2026-11-13 (§169, §205, §211, §214, §216): still RETRACTED; one lever has narrowed the
gap and none has closed it.** Partitioned contexts (a depth-2 tree, each leaf with its own
2,000-row context) lift zero-shot mean AP from 0.1986 to **0.2362** on the five full-panel
folds (§169, 5 of 5 folds positive, 4 of 5 significant after Holm). That closes 19% of the gap
to untuned CatBoost's 0.4005 (§169's own figure); CatBoost is still 1.70 times FinTFM's AP. It
also beats §163's fine-tuned arm (0.2062) on 5 of 5 folds (+0.0299, §205). No other inference
lever has cleared the bar since: §211 records context composition as exhausted and muon as
harmful. Outside V4FinBench, FinTFM is below untuned logistic regression on four credit
datasets (§216). Systems that put FinTFM's logit into a booster are not FinTFM accuracy: a
capped tuned LightGBM gains +0.0304 mean AP with it (0.3662 to 0.3967, §214), which stays below
uncapped CatBoost's 0.4005.

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

**Status: SINGLE DRAW (owner decision, 2026-11-13).** Each of the four panels is one 70/30 split with no seeds, the evidence is on the pre-cell-attention architecture, and the published checkpoint misses the default rate (§174 and the DR4.3 run record), so the agreement of four single draws is directional, not a surviving claim. Re-tagging needs ECE on the published checkpoint over five splits of the credit suite. Earlier tag, kept: **Status: SURVIVES, on the pre-cell-attention architecture.** (Re-read 2026-11-13: the caveat
below, one 70/30 split per dataset and no seeds, means each panel is a single draw by this
ledger's own vocabulary, and the four panels agreeing is what keeps the tag. Quote it as
directional; flagged for the owner.) Evidence: §73, four real panels
outside V4FinBench (Polish bankruptcy at three horizons, Taiwan bankruptcy), against classical,
random forest and gradient-boosting baselines (untuned, per §73's caveats).

fintfm has the **best or near-best ECE on all four panels**, independent of which prior wins on
discrimination there (§75 found the prior that loses on V4FinBench, pure SCM, wins
discrimination on all four). (Corrected 2026-10-15: this entry previously listed V4FinBench
among the four panels, which §73's ECE table does not include, said "tuned" baselines, and
said "three of the four".)
This is the first time the calibration thesis (§12) has been tested outside V4FinBench, and it
held without exception, while the discrimination gap to boosters varies **1.7x to 7.2x** across
the same four panels at similar base rates — not explained by class balance, so it tracks
dataset structure specifically.

**This is arguably the strongest surviving empirical claim in the project besides Claim 1.**
It is also the one whose mechanism is least understood: why calibration transfers when
discrimination does not is an open question this ledger does not yet have an answer to, and it
connects directly to task 39.7's proposed task-representation probe (does the model form a
task posterior, separate from how well that posterior discriminates?).

**Not shown for the published (cell-attention) checkpoint (noted 2026-11-09).** No ECE comparison has been run
for it on the credit suite. Its mean predicted default level misses the observed rate on ULB (0.212% to
0.300% against 0.173%, §174, where logistic regression gives 0.167% to 0.177%) and on FDIC splits 1 and 2
(1.9 and 3.2 times, below it on split 0; these three ratios are in the DR4.3 draft
`docs/results/drafts/dr4.3-runs_2026-11-07.md`, and §216 does not repeat them). On SBA 7(a) the
level is right, 3.2% to 4.0% predicted against 3.9% to 4.4% observed (§177). This does not refute the claim, which is scoped to the
older architecture, but nothing may extend it to the current model until the ECE comparison is run.

**Caveat**: single 70/30 split per dataset, not the five-fold paired-bootstrap standard the
V4FinBench numbers meet (§69, itself on a 10x subsample of 105,900 test rows, §82). Directional, not settled at this precision.

---

## Claim 10 — Two-way cell attention is a real, reproducible architectural fix for a severe capacity defect

**Status: SURVIVES on synthetic data and on one real panel (V4FinBench), at matched
context.** Evidence: §78 (synthetic), §80 (V4FinBench, five folds). (Updated 2026-10-18,
§172: the V4FinBench half now also holds at a matched prior and protocol, one training seed
per arm.) §78 is one
experiment (`n_cell_blocks=1`, `cell_labels=True`, both `p_financial` arms), confirmed by two
independent instruments (a Bayes-ceiling probe with a closed-form, numerically-verified target,
and the pre-existing antisymmetric probe).

**Narrowed 2026-10-16 (audit task AU2.8).** The real-panel half compares two checkpoints that
differ in prior mix as well as architecture (`p_financial` 1.0 against 0.7, §80's confound
note). Until a matched pair is trained, quote the V4FinBench gain as "cell attention plus a
change of prior mix", not as an architecture effect. The synthetic half (§78) is unaffected.

**Resolved 2026-10-18 (§172).** The matched pair has been trained: both arms re-trained at
seed 0, 48,000 tasks, fp32, `p_financial` 1.0, differing only in `--n-cell-blocks 0
--no-cell-labels`, scored zero-shot on the five full-panel folds at the packaged default
inference settings. The cell-attention block and its labels are worth **+0.0520 mean AP**
(0.1947 against 0.1428), lower without them on 5 of 5 folds, Holm p 0.010 to 0.012 on a
500-resample company-level bootstrap. The V4FinBench gain can now be quoted as an architecture
effect at the default context. Gaps: one training seed per arm; the two arms were scored on
different devices (CPU and GPU) over identical test rows; §84's best-vs-best comparison has not
been repeated at a matched prior. §80's +0.0486 and §84's +0.0417 below are kept as the
mismatched-prior history.

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
own, now fixed) and a second seed. (Superseded 2026-10-15: §86 trained the full protocol on
a T4, and §142 replicated §91's matched pair, which includes the cells-plus-labels arm, at
seed 1.) One honest anomaly is on the
record rather than smoothed over: `symmetric_count` is slightly lower on both new checkpoints
than their old-architecture counterparts, unexplained.

**Task 39.5 closed, 2026-09-15 (§80): the fix transfers.** Two complete five-fold V4FinBench
runs on the full 1.0M-row panel, differing only in `n_cell_blocks`/`cell_labels`, give cell
attention **+0.0486 mean AP, 5 folds of 5, every 95% CI excluding zero, every Holm-corrected
p < 0.001** (possibly confounded with the prior mix, audit task AU2.8; **confirmed
2026-10-16**: the cell arm is `v4-cellattn-fin10`, `p_financial` 1.0, and the old arm
`v4-colid-fin07`, `p_financial` 0.7, per §80's confound note, and no matched pair was ever
trained, so the architecture-only effect is unmeasured; the same applies to §84's +0.0417 and
§91's reference arm; **resolved 2026-10-18 (§172)**: a matched pair at `p_financial` 1.0 gives
+0.0520 mean AP, 5 of 5 folds, see above). The synthetic result
did not dissociate on real data — the specific failure §47
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

**Reviewed 2026-11-13: tag unchanged (§172, §195, §202).** The matched-prior pair of §172
(+0.0520 mean AP, 5 of 5 folds) still stands at one training seed per arm. Nothing since adds a
second architecture comparison; the later scale arms keep the cell-attention architecture and
change size. At about twice the parameters, neither depth (8 layers, §195, §193: -0.0047 mean AP,
3 of 5 folds positive) nor width (192, §202: +0.0003 mean AP) moved V4FinBench at 480,000 tasks,
so the architecture is not obviously capacity-limited on credit at this volume either.

---

## Claim 11 — Measured externally, this is not a competitive general tabular model

**Status: SURVIVES.** Evidence: §98 (placement), §100 (decomposition). This is the only claim
in this ledger whose baselines and protocol belong to someone else.

TabArena, 27 eligible datasets, zero failures: **rank 93 of 95 [§225: read from a table numbered from 0; §98's own leaderboard is not stored]**, mean ROC-AUC **0.7642** at
§98, improved to **0.7823** by native categorical handling (§101) **with the rank unchanged**.
**On the later board snapshot used for the submission the rank is 94 of 95** at the same
0.7823 (§122), and a refreshed baseline cache left it at 94 of 95 (§150). Quote 94 of 95.
Below every linear baseline; above only default KNN.

**That a +0.0181 mean gain moved the rank by zero is itself the finding.** Near the bottom of
a 95-method leaderboard the Elo spacing is wide — the next method up is 94 Elo away — so rank
is a coarse instrument and quoting it alone would have hidden a real improvement. Cite the
mean and the rank together, never the rank alone.

**Why a negative result is entered as a surviving claim.** Every accuracy number produced
before §98 was measured against baselines this project ran itself — a closed loop, which
cannot detect a harness-level error that flatters every arm equally. This breaks the loop and
confirms the direction the internal measurements already pointed. A project whose public
record contains only its own benchmarks has not been tested; this one now has.

**The "genuinely favourable detail" was withdrawn in §107 and must not be reinstated.** This
claim previously noted that two of the three best results were the corporate-bankruptcy panels
the model was designed for (`taiwanese_bankruptcy_prediction` 0.9291,
`polish_companies_bankruptcy` 0.8436). Those are **absolute** AUCs compared against nothing.
By rank among the 95 methods they are **83rd and 91st**; `GiveMeSomeCredit` is 91st, `heloc`
94th and `credit-g` 95th. Every other method scores well on Taiwanese bankruptcy too. **High
absolute AUC on a credit panel is not competitiveness on that panel**, and the earlier
paragraph made the error it explicitly disclaimed.

**Nor is there a specialist profile.** TabArena reports mean rank 86.26 against harmonic rank
20.72 — the largest spread of any of the 95 methods — which reads as "near-last typically,
near-top where it belongs". It decomposes to **one** dataset at rank 1, one at 33, one at 77,
and 24 in the 83–95 band; **median rank 94**. The rank-1 is a single fold with a 0.0026 AUC
margin. A harmonic mean is dominated by its minimum, and here the minimum is n = 1 (§107).

**Scale is not the explanation, and this is now measured rather than argued** (§114). A 5.0M
model at *matched* task volume scores −0.0049 against the 885K baseline; training volume is
null at 5× and +0.0028 at 2×; a peer's published curve returns +0.0049 R² for 16.7× the
parameters; and the deep-narrow alternative could not complete the benchmark. A reader will
reach for "the model is small" — it is quantitatively insufficient by roughly an order of
magnitude.

**Four bounds a citation must carry.** Coverage was 51% at §98 (26 of 51 scored, binary-only,
≤136 features), briefly 90% while multiclass and regression were declared, and is **53% now**
(27 of 51) after §121 withdrew both — they rank last. The three figures are not comparable, so
each number carries its own. The categorical path was knowingly degraded at §98, which §100
quantifies and §101 fixes. One earlier run died of `TimeLimitExceeded` from a CUDA-only device
check on Apple Silicon — an integration defect, not evidence about deployability. And
**TabArena-Lite runs fold 0 only**: any per-dataset statement from it is a single draw and must
be labelled as one.

**Superseded 2026-10-15 (§122):** the binary-only submission described below was made as
autogluon/tabarena#616 and declined on competitiveness. A TabArena maintainer confirmed the
integration and numbers correct, and also judged a binary-only declaration to be selective, so
any future submission needs all 51 datasets. The paragraph is kept as first written.

**The PR conditions are now met, by losing an argument with the measurement.** The categorical
path is target statistics (§101); multiclass and regression were declared, scored, and
**withdrawn** when §121 measured them ranking last. The submission declares `binary` only at
53% coverage — narrower than the 90% entry, and true. Weights are public and Apache-2.0, so the
number is reproducible.

**Updated 2026-11-13 (§204, §220-§224): tag unchanged; the zero-shot figures above stand and a
second, different entry now exists.** The zero-shot default on the 27 binary datasets scores 756
Elo, 94th of 95 (the 2026-10-19 run, §204). The all-51-dataset entry that DR7.6 requires is a
different model configuration ("FinTFM with a trained linear readout", Claim 19), not an update
of this one, and it ranks 107th of 110 at 903 Elo (§224). Read the two rows as separate
entries. The "Four bounds" paragraph still applies to every TabArena-Lite figure: fold 0 only.

---

## Claim 12 — The multiclass head learns in context, and inherits the same deficit

**Status: SINGLE DRAW.** Evidence: §99. One checkpoint, three seeds per cell, one probe
family. Not quotable externally until replicated on a second seed and a second task shape.

A checkpoint trained at `--max-classes 10` clears an untrained control of the identical
architecture at every class count tested — **+0.261 / +0.267 / +0.157 macro one-vs-rest AUC at
K = 3 / 5 / 10** — while the control sits at chance throughout (0.471–0.527). Before this, "the
model does multiclass" was a statement about a command-line flag; the capability suite was
binary-only and had no instrument to check it.

**The deficit to a correctly-specified baseline is large and widens with K**: 0.211 → 0.262 →
0.359 macro AUC behind multinomial logistic regression. That is the same shape §96 found for
binary tasks, so **multiclass inherits this project's central open problem rather than
introducing a new one** — which is the useful part of the result, and the part a summary would
be tempted to drop.

**A measurement detail worth keeping**, because it would have inflated the margin: the probe's
classes are not exactly balanced (a random argmax partition gives 4.4%–15.3% at K = 10), so the
accuracy floor is the *measured* majority-class rate, not `1/K`. Using `1/K` would have
understated the floor by 0.06 at K = 10.

**Reviewed 2026-11-13 (§181, §186, §197, §198, §221-§224): tag unchanged, scope of the evidence
widened.** §99 is still one checkpoint on one probe family, so it stays SINGLE DRAW. Real
multiclass tasks now give a second, different shape: on the proxy suite (6 of 8 scorable tasks,
3 splits, an unvalidated benchmark, §216), `dr74-mc-s0` (class permutation at 48,000 tasks) beat
the earlier multiclass checkpoint on 6 of 6 tasks at seed 0 (§186) and on 5 of 6 at seed 1 (§197,
candidate under the amended per-split criterion), reaching 994 and 1171 Elo against RF at 1,000.
Ten times the tasks was null (§198, §202). On TabArena-Lite (8 multiclass datasets, fold 0) the
linear-readout entry reaches 865 Elo at best (§224), against 306 Elo for the default entry
(§221). Multiclass is still far behind the best baselines on the four `mfeat` views (§186), so
the widening deficit with K in §99 has not been disproved.

---

## Claim 13 — Most of the external deficit is preprocessing, and it is fixable without retraining

**Status: SURVIVES as a diagnosis; the *size* is a third of what the bound suggested.**
Evidence: §100 (decomposition), §101 (full suite, 27 datasets).

fintfm embeds every cell as a numeric scalar, so a categorical column must arrive as a number.
Label encoding asserts an order that does not exist. Replacing it with **out-of-fold smoothed
target statistics** — computed from context labels only, so no gradient step and no retraining
— moved `Amazon_employee_access` from **0.5455 to 0.8220** ROC-AUC (§101), within 0.022 of
tuned logistic regression (no FINDINGS source found 2026-10-15; the figure appears only in
`docs/CHANGELOG.md`), on the identical checkpoint. Two lower-cardinality datasets gained
+0.034 and +0.017 (no FINDINGS source found 2026-10-15; §101's full-suite run lists
`Diabetes130US` +0.0618, `customer_satisfaction_in_airline` +0.0400 and `credit-g` +0.0344),
ordered exactly as §100's cardinality correlation predicts.

**Two controls that make the A/B readable**, both of which held. The label arm reproduced
§98's stored scores to four decimal places, confirming checkpoint identity and determinism, so
the delta is the encoding alone. And the result lands *below* the baselines rather than above
— a leaking target encoder would have scored above 0.95 on a column of near-unique levels.

**Measured on the full suite (§101), and the diagnosis held while the magnitude did not.**
Mean ROC-AUC 0.7642 → **0.7823**, gap to tuned logistic regression −0.0527 → **−0.0346**. The
decisive number is not the mean: **the correlation between the per-dataset gap and log maximum
cardinality collapsed from −0.668 to −0.025**, and the eight numeric-only datasets moved by
exactly 0.0000. A mean can improve for many reasons; a destroyed correlation is specific to
the claimed mechanism.

**§100's 82% was a ceiling and the realised figure is 34%.** That entry said so in advance —
"a ceiling, not a forecast" — but the gap between them is recorded here because a bound quoted
without its outcome gets read as a prediction later.

**The real prize was structural.** The suite used to hold two populations: a −0.0320 modelling
deficit on numeric data and a −0.0894 preprocessing artefact on categorical data. It now holds
one, near-flat at **−0.0320 / −0.0351 / −0.0364 / −0.0379** across every bucket. Any future
account of the residual must explain a uniform ~0.035 deficit independent of categorical
content, cardinality and width — a far sharper target than §98 presented.

**What this claim must not become.** It is not a route to competitiveness. **The rank did not
move: 93 of 95 [§225: one place lower], unchanged** (94 of 95 on the later board snapshot, §122), because a uniform −0.035 still loses to the methods above.
`RF (default)` leads on 21 of 27 datasets. The stated target — untuned trees < fintfm < tuned
trees — is not reached.

---

## Claim 14 — Regression and multiclass are real capabilities, not declarations

**Status: OPEN.** (Retagged 2026-11-13, §121, §221-§224: **SINGLE DRAW** as a capability claim.
Both run correctly on real TabArena-Lite data and are not competitive; every real-data figure is
fold 0 only, and no credit regression or multiclass panel has been scored. The original text
below is kept.) Evidence: §99 (multiclass on synthetics), §105 (multiclass costs binary
nothing), §106 (regression on synthetics). Open because no real-data regression or multiclass
number exists yet. (Superseded 2026-10-15: §121 scored both on real TabArena data, 7
multiclass and 12 regression datasets. Both run correctly and rank near last, mean rank 93.4
and 93.1, so they are working capabilities and not competitive ones. No credit regression
panel, LGD or EAD, has been scored.)

What is measured:

- **Coverage 51% → 90% → 53%** (27 of 51), derived at run time from TabArena's task
  metadata **and the model's own declaration**, never asserted. The last figure is after
  §121 withdrew multiclass and regression for ranking last; `max_features=136` excludes the
  remaining 3 binary datasets.
- **The mixed prior is free.** A `--max-classes 10` checkpoint with the generic SCM prior
  mixed in scores **0.7817** against the binary-only checkpoint's **0.7823** on the same 27
  binary datasets — paired mean delta **−0.0006**, better on 13 of 27, sign test **p = 1.000**
  (§105). Capacity spent on 10-class structure is not capacity taken from the binary task, at
  885K parameters.
- **The binned head regresses** (§106). The untrained control of the same architecture sits at
  the predict-the-mean baseline on all three target shapes; the trained model reaches
  0.34–0.72 nRMSE, beats ridge where ridge is misspecified, and produces 80% intervals
  covering 0.771–0.847 untuned.

What is **not** measured, and why the claim stays open: no real regression panel, and
TabArena's 12 eligible regression datasets are unscored. (Superseded 2026-10-15: §121 scored
them; mean rank 93.1, last on 5 of 12.) §106's evidence is synthetic probes
with known structure.

**Real-data numbers, 2026-11-13 (§221-§224; TabArena-Lite, fold 0, CPU; the entry is "FinTFM with a
trained linear readout", Claim 19).** Multiclass, 8 datasets: 751 Elo with a probe per partition
leaf (§221), 825 with a scaled, cross-validated readout plus the row's own features (§222,
better on 8 of 8, sign test p 0.01), 865 with four partition bags (§224, better than §222 on 5
of 8). Regression, 13 datasets: 436 Elo with a ridge readout (§220), 578 with the row's own
features added (§223, better on 12 of 13). These are readout results, not the zero-shot head, and
both types still rank near the bottom of the leaderboard (overall 107th of 110, §224).

**The trap this claim exists to flag.** On a bimodal target, an *untrained* model scores
**higher** outer-third mass (0.8628) than the trained one (0.8083) and the truth (0.8362),
because quantile-bin representatives are themselves concentrated at the ends. A distributional
claim supported by outer mass alone would be measuring the binning. Only the conjunction with
per-row assignment (Spearman 0.658 against the control's 0.055) carries it.

---

## Claim 15 — The prior mixture had no member generating axis-aligned structure, and now it does

**Status: OPEN**, with one component **RETRACTED** and the headline result **bounded to
general tabular data**. §116 measures the tree prior at **−0.0221 AP on V4FinBench** — the
low-default credit panel this project exists to serve — against +0.0094 on TabArena. Both hold.
`p_tree` stays 0.0 by default and the prior is not shipped. (§166, 2026-10-16: the −0.0221 is
not a measurement of the tree prior. Because the prior draws the tree source first, §116's tree
arm trained on financial 0.49 against its control's 0.70; against `scm50`, a control at the same
financial share, the tree arm is +0.0119 mean AP, 4 of 5 folds positive, one seed per arm and a
row bootstrap. The credit result is open, pending task DR2.4's re-test at a matched share and
two seeds.) What survives is the *instrument*:
§112 predicted a general-tabular gain from a distinctiveness statistic and §115 confirmed it at
two seeds, which is the first prior-side prediction this project has had confirmed.
**Distinctiveness predicts breadth, not fit** — a criterion taken from a general-tabular paper
optimises for general tabular data. Evidence: §111 (retracted in part), §112
(the correction and the standing numbers). Open because no checkpoint trained with the tree
prior has been scored yet — the arm is training. (Superseded 2026-10-15: §115 and §116 scored
tree-prior checkpoints at two seeds, with the results stated at the top of this claim.)

**The retraction, stated before the claim.** §111 reported that this project's SCM prior
"generates tasks a *linear* model wins on", measured as a −0.0234 tree-versus-linear AUC gap.
**That number is wrong and the claim built on it is withdrawn.** Its linear baseline was fitted
on raw heavy-tailed features and failed to converge on eight of twenty-five tasks; since the
statistic is `tree_auc − linear_auc`, understating the linear arm inflated it. Corrected, the
SCM prior's gap is **−0.0021** — no effect. The SCM prior is *neutral* on this axis, not
linear-favourable.

**What the corrected measurement supports** (`fintfm-priorscore`, 25 tasks per prior requested, 21 scored for `scm` per §112, all
statistics from *fitted* baselines rather than this project's model):

| prior | performance | diversity | distinctiveness |
| --- | --- | --- | --- |
| financial | 0.6880 | 0.1343 | −0.0292 |
| scm | 0.7106 | **0.2046** | −0.0021 |
| **tree** (new) | **0.7555** | 0.1065 | **+0.0225** |
| *trivial (control)* | *0.9968* | *0.0024* | *−0.0142* |

`tree` is the **only** prior with positive distinctiveness. Both incumbent priors sit at or
below zero, so the mixture contained no member generating the axis-aligned, piecewise-constant
structure that every tree ensemble above fintfm on TabArena is built to exploit. That is the
claim, and it is a statement about the *prior*, not about accuracy.

**Two secondary readings that change what should be done.** The SCM prior has by far the widest
difficulty spread (diversity 0.2046 against 0.107–0.134) — the property §42 established this
project needed — so its weight should not be cut to make room for the tree prior. And the
financial prior is the **hardest** of the three at 0.6880, consistent with §96 rather than in
tension with it.

**Why this claim cannot yet be upgraded.** Distinctiveness is a necessary condition for
structure to be learnable in context, not a sufficient one. §93 measured a 5× training-volume
increase as inert, and a distinctive prior can be inert the same way. Only the paired arms —
`p_tree` against the current mixture at matched tasks, on V4FinBench and TabArena — decide it.
(Superseded 2026-10-15: those arms were run in §115/§116: +0.0094 mean ROC-AUC on TabArena,
−0.0221 AP on V4FinBench, so the tree prior is not shipped.) (§166, 2026-10-16: the V4FinBench
arm was not matched; it also lowered the financial share from 0.70 to 0.49. At a matched share
the tree arm is +0.0119 mean AP, one seed, so these arms did not decide the credit question.)

**Reviewed 2026-11-13 (§180, §191): tag unchanged; the credit question is still open.** DR2.4 as
run did not re-test the tree prior at a matched financial share. It screened prior v2 (financial
v2 alone, a general trunk alone, trunk then financial, and the two interleaved) at 48,000 tasks
and seed 0. All four were harmful on V4FinBench against `e1-v48-s0`'s 0.1975 mean AP, each with 0
of 5 folds positive (-0.0212, -0.1262, -0.0546, -0.0358, §191); on the proxy suite the trunk arms
scored highest (764 Elo for trunk-then-financial against 548 for the control, §191). The
general-tabular and credit results disagree again, as in §116, and the proxy is not validated
(Claim 18). A matched-share, two-seed tree-prior comparison on V4FinBench has not been run.

**The methodological failure is part of the claim, not a footnote.** The first version of this
instrument reported the financial prior as the *most* tree-favourable member (+0.0754), which
would have made the tree prior largely redundant and was the strongest available argument
against building it. That argument was an artefact of the diagnostic's own preprocessing —
and the defect was one this project had already measured, published and fixed on the inference
side four weeks earlier (§35, rank conditioning, worth +0.086 AUC). A weak baseline in a
benchmark flatters the model and someone notices; **a weak baseline in a diagnostic silently
becomes the measurement**, and +0.0754 for a prior full of accounting thresholds is plausible
enough to pass review. Any paper reporting prior-selection statistics must state the baseline's
preprocessing explicitly.

---

## Claim 16 — Opt-in dataset-specific fine-tuning raises V4FinBench AP, by a small margin

**Status: SINGLE DRAW** (one fine-tuning seed per fold). Evidence: §163. A replication and two
variants are pre-registered as §164, with no result yet. **This is a claim about a fine-tuned
model, not about the zero-shot FinTFM**, and it does not carry Claim 2's provenance argument
(decision D18).

**What was measured.** `FinancialTFMClassifier(finetune_steps=200)` (0.7.0, off by default)
takes 200 AdamW steps at lr 1e-5 on a copy of `runs/v4-cellattn-labels.pt`, on tasks built from
each fold's training rows only. Five folds of the full 1,000,087-row V4FinBench horizon-0 panel,
paired bootstrap against the same checkpoint zero-shot (`control_rescore.json`), Holm across
folds:

| fold | 0 | 1 | 2 | 3 | 4 | mean |
| --- | --- | --- | --- | --- | --- | --- |
| zero-shot AP | 0.2113 | 0.2050 | 0.1786 | 0.1810 | 0.2172 | 0.1986 |
| fine-tuned AP | 0.2181 | 0.2068 | 0.1901 | 0.1955 | 0.2205 | **0.2062** |
| ΔAP | +0.0068 | +0.0018 | +0.0115 | +0.0145 | +0.0033 | **+0.0076** |
| Holm p | <0.001 | 0.291 | <0.001 | <0.001 | 0.138 | |

The mean is the mean of the five per-fold APs. **5 of 5 folds positive, 3 Holm-significant.**
By §163's pre-registered rule (mean ΔAP above +0.0073 with at least 4 of 5 folds positive) the
verdict is *helps*, and the prediction held. It is the first lever to move V4FinBench AP after
D16's seven, real-data pretraining (§160-§161) and context construction (§162).

**Why this is narrower than "+0.0076".**

- **One fine-tuning seed** per fold. Fine-tuning's own seed variance is unmeasured, and the
  paired bootstrap resamples rows, not fine-tuning runs, so its intervals cannot see it (§158).
- **The bar it cleared is not a noise estimate.** The 0.0073 mean AP yardstick is §159's, and
  §160 showed it is mostly a head-size effect: the `max_classes 10` matched control sits −0.0082
  mean AP below the `max_classes 2` control (5 of 5 folds), while two `max_classes 10` controls
  at seeds 0 and 1 differ by +0.0009. So the margin over the bar, 0.0003, says nothing about
  whether the effect exceeds noise; the 5-of-5 sign consistency is the stronger evidence, as
  §163 itself says (audit task AU2.9).
- **The learning rate was chosen on a tie.** Stage A's fold-0 validation AP was 0.23247 at both
  1e-5 and 3e-5.

**Against the gradient-boosted trees (§163, same five full-panel folds, boosters untuned at
library defaults).**

| model | setting | mean AP, 5 folds |
| --- | --- | --- |
| CatBoost | 1,000 iterations | **0.4005** |
| LightGBM | 1,000 trees at lr 0.05 (§167, added 2026-10-16) | **0.3908** |
| LightGBM | 100 trees, under-trained (§167) | 0.2666 |
| XGBoost | 100 trees, under-trained (§167) | 0.2229 |
| FinTFM, fine-tuned | 200 steps | 0.2062 |
| FinTFM, zero-shot | | 0.1986 |
| logistic regression | untuned | 0.1614 |

Untuned CatBoost wins every fold, at 1.94× fine-tuned FinTFM's AP (a 0.194 AP gap). **Fine-tuning
closes 3.8% of the zero-shot gap to CatBoost.** Fine-tuned FinTFM is within 0.017 AP of untuned
XGBoost and 0.060 of untuned LightGBM. The published protocol tunes the boosters, but its
`--tune` grid caps CatBoost at 200 iterations, so tuning under that grid could lower CatBoost
rather than widen the gap (§163, corrected 2026-10-16). (§167, 2026-10-16: every row uses
V4FinBench's published features, which keep `operational_status`. LightGBM at 1,000 trees comes
within 0.0097 of CatBoost, so the lead is boosting rounds; "within 0.017 AP of XGBoost and 0.060
of LightGBM" compares with under-trained 100-tree defaults and is not a distance to the field,
which is about 0.39-0.40 mean AP. Without `operational_status`, CatBoost's defaults score
0.3662; FinTFM's dependence on the column is unmeasured, task DR3.10.)

**Pending: §164, pre-registered 2026-10-15.** Three arms, each one change from §163's setting:
`S1` (`finetune_seed: 1`; fine-tuning's seed noise, and whether §163 replicates), `EN`
(`finetune_query_pos_frac: 0.1`, importance-weighted) and `L600` (600 steps). `EN` and `L600`
run on a Kaggle GPU, so their verdicts stay provisional (device-confounded) until §163's own
setting is re-run on the same kernel. Nothing from §164 may be quoted until it is in FINDINGS.

**Resolved 2026-11-13 (§164, closed 2026-10-17; §202, §208, §192).** §164 is in FINDINGS and the
status stays SINGLE DRAW, now with a second fine-tuning seed that agrees in direction.

| arm (change from §163's setting) | mean AP | against §163's 0.2062 | verdict |
| --- | --- | --- | --- |
| `S1`, `finetune_seed: 1` (CPU) | 0.2057 | -0.0005 | against zero-shot +0.0071, 5 of 5 folds positive; **not replicated by §163's rule**, which needs above +0.0073, missed by 0.0002 |
| device re-run of §163's setting (GPU) | 0.2063 | +0.0001 | device gap negligible |
| `EN`, 10% default queries, importance-weighted (GPU) | 0.2071 | +0.0009 | no different (3 of 5 folds) |
| `L600`, 600 steps (GPU) | 0.2086 | +0.0024 | no different (3 of 5 folds) |

Fine-tuning's seed noise is small (mean absolute per-fold difference 0.0007), so the effect is
consistent across the two seeds; what failed is the size threshold. None of the three stronger
settings improves on it. Three later results narrow the claim further. First, §169's zero-shot
partitioned contexts reach 0.2362 mean AP, beating §163's fine-tuned arm on 5 of 5 folds
(+0.0299, §205), so fine-tuning is no longer the largest lever. Second, fine-tuning per leaf of
that partition is null against the partition alone (+0.0049 mean AP, 4 of 5 folds, §208). Third,
dropping `operational_status` costs the fine-tuned model -0.0023 mean AP (§202), so the gain is
not carried by that column. On general tabular data, 200 fine-tuning steps per fit beat the
default on 16 of 16 proxy binary tasks (§192, one split per task, A10G; a proxy result, not a
TabArena or credit claim). The "Write it as" sentence above is unchanged.

**Write it as:** "opt-in fine-tuning per dataset raised mean AP from 0.1986 to 0.2062 on five
full-panel V4FinBench folds, 5 of 5 positive, at one fine-tuning seed; the fine-tuned model has
seen the dataset's training rows." Never as a property of the zero-shot model, and never
without the CatBoost row beside it.

---

## Claim 17 — Real-data pretraining (FinTFM-R) has not improved V4FinBench AP at the one recipe tried

**Status: SURVIVES as a null result, at one recipe** (30% real share, 6,000 steps from scratch,
two seeds per source). **Family: FinTFM-R.** Evidence: §157 (CPU pilot), §160 and §161 (GPU,
five folds, full 1,000,087-row panel). Decision rules pre-registered in §159 and applied by
`scripts/c5_verdict.py`.

Each arm against the `max_classes 10` control at the same seed, mean AP difference over five
folds:

| source | seed 0 | seed 1 | §159 Rule 1 |
| --- | --- | --- | --- |
| EDGAR | −0.0019 | −0.0029 | null |
| SBA 7(a) | −0.0355 (5 of 5 folds negative) | +0.0024 | null (seed means disagree in sign) |

- **EDGAR is a stable null.** Its two seeds differ by −0.0002 on the mean, as close as the two
  no-change controls (+0.0009, §160).
- **SBA is a null whose seeds disagree.** SBA seed 1 against SBA seed 0 is +0.0388 mean AP, 5 of
  5 folds. Adding the SBA panel turned a run whose outcome barely depended on the seed into one
  where the seed moves AP by about 0.04. Two seeds cannot say which is typical. PPDai was not
  launched, under §159 Rule 2.
- **The CPU pilot did not predict the GPU result.** §157 continued pretraining from
  `runs/v4-cellattn-labels.pt` for 1,500 steps, one arm per source (six sources), scored on fold
  0 at the 200k-row development cap (34,906 test rows, 130 positives): no arm separated from the
  control after Holm, SBA's lead shrank to +0.0073 mean AP over five folds (3 of 5 positive, none
  significant), and the pre-registered base-rate hypothesis was rejected. §162 later found the
  cap can change conclusions, so the pilot's ranking may not transfer; §160-§161 are full-panel.
- **Training loss carries no information about transfer here** (§161): both SBA seeds and both
  EDGAR seeds fit their mixtures better than the control and transfer no better.

**What this decides.** D17's premise, that real financial data improves the credit benchmark,
has no support at this recipe. Other shares, step counts and sources are untested, so this is
not evidence that real data cannot help. No FinTFM-R result may be described as an
improvement.

**Peer note 2026-11-06 (`docs/paper/RELATED_WORK.md`, "Peer read of 2026-11-06").** FTFM
(arXiv:2610.25558) reports, for one architecture on TabArena Lite, a synthetic-only stage at mean
error 0.1692 against 0.1620 once real tables and larger tables are added (as read from its Table 3;
real data is confounded with table size). That is a different mix, scale and benchmark from the
recipe above, so Claim 17 stays a null at one recipe, but "real data cannot help" is also not
supported by the field.

---

## Claim 18 — Four training ablations and context size do not move V4FinBench AP

**Status: SURVIVES as a negative result**, on one checkpoint for the context question. Family:
FinTFM (synthetic-only). Evidence: §158, §160 (re-adjudication), §162 (context).

**Four Kaggle ablations, re-scored against a matched control (§158, §160).** §144-§147 trained
at `max_classes 10` and had been scored against a `max_classes 2` control, so each difference
held a head-size and initialisation change beside the change under test. Against the matched
`max_classes 10` control at seed 0 (five folds, full panel, mean AP difference, §159 Rule 3):

| ablation | mean | folds negative | verdict |
| --- | --- | --- | --- |
| §144 realism | +0.0047 | 1 of 5 | null |
| §145 learnability | −0.0024 | 3 of 5 | null |
| §146 schedule-free | −0.0098 | 5 of 5 | loss |
| §147 mask embedding | −0.0230 | 5 of 5 | loss |

Code drift between the ablations' commits and the control's was ruled out for the task sampler
and the initialisation (identical hashes across six commits, §158).

**Context size is not what limits V4FinBench AP (§162).** A fold-0 diagnostic at the 200k-row
development cap (34,906 test rows, 130 positives, a training pool of ~106k rows) had a 500-row
context beating the 2,000-row default, 0.2215 AP against 0.1392. The pre-registered five-fold,
full-panel confirmation failed: a 500-row context with 8 ensemble members scored −0.0029 mean AP
against the default (2 of 5 folds positive), and a single 500-row draw −0.0404 (1 of 5). The cap
changed the shape of the context curve, not only its level: the default scores 0.1392 on fold 0
at the cap and 0.2113 on the full panel. The scoring default stays.

**What this establishes.** The development cap distorted §162's fold-0 context conclusion, and
the untuned-logistic-regression lead it showed (0.1658 against 0.1442 AP) does not exist on the
full panel (Claim 6, update of 2026-10-16). The remaining gap is to the gradient-boosted trees,
and fine-tuning (Claim 16), the one lever since D16 to move AP, closes 3.8% of it.

**Extended 2026-11-13 (§188, §195, §202, §211; family FinTFM, synthetic-only).** The negative
result now covers more levers, each on V4FinBench's five full-panel folds with the +0.0117 mean
AP yardstick (§173) and at least 4 of 5 folds positive as the bar. The last sentence above is
out of date: fine-tuning is no longer the only lever to have moved AP, because partitioned
contexts did (Claim 6, §169).

| lever | result | verdict |
| --- | --- | --- |
| ten times the pretraining tasks (480,000 against 48,000), two seeds | -0.0109 and -0.0103 mean AP, 1 and 0 folds positive (§188) | null, both seeds |
| 8 layers at 480,000 tasks (§193, §195) | -0.0047 mean AP, 3 of 5 folds | null |
| width 192 at 480,000 tasks (§202) | +0.0003 mean AP | null |
| k-means prototype context (AU4.2, §211) | +0.0002 | null |
| `max_context` 900 with 3 members (AU4.3, §211) | +0.0093, 4 of 5 folds | null |
| rank plus squash member transforms with 8 members against 8 plain members (AU5.12, §211) | +0.0038 | null |
| longer training contexts (AU5.4, §211) | -0.0042 | null |
| `--optimizer muon` (AU5.17, §211) | -0.0125, 0 of 5 folds | **harmful** |
| per-class retrieval, 16 positives (AU5.8, §213) | -0.1934, 0 of 5 folds | **harmful** |
| RBF value embedding (AU5.16, §213) | -0.0321, 0 of 5 folds | **harmful** |
| fixed positive floor in the context (DR3.3, §200) | +0.0024 (exact), +0.0055 (floor16) | null |
| 8 members (§196) | +0.0058, 3 of 5 folds | lead, not adopted (half the bar) |
| fine-tuning per partition leaf (§208) | +0.0049 against the partition arm, 4 of 5 folds | null |

Only the partition is a candidate (+0.0375, §169). Two cautions. The rows are paired against
different controls (the zero-shot default 0.1986 for inference arms, the same-seed 48,000-task
control 0.1975 for pretraining arms, §211), and each pretraining arm is one seed except §188.
And §188's volume result is credit-specific: on the unvalidated proxy suite, 480,000 tasks beat
48,000 by +168 and +186 Elo at the two seeds (§188), and DR4.2 found the proxy does not rank
checkpoints as TabArena-Lite does (Spearman +0.717 against the 0.8 bar, §216). Volume is not a
credit lever; whether it is a general-tabular one is unmeasured on a validated benchmark.

---

## Claim 19: "FinTFM with a trained linear readout" reaches 903 Elo on all 51 TabArena-Lite datasets, 107th of 110, and is a general-tabular result

**Status: SINGLE DRAW** (TabArena-Lite runs repeat 0, fold 0 only; one checkpoint per problem type).
Family: FinTFM (synthetic-only). Evidence: §212, §215, §220-§224. Added 2026-11-13. **This is a
general-tabular result, not a credit result**, and it is not the zero-shot model: a logistic
regression (binary, multiclass) or a ridge regression (regression) is fitted on the network's
pre-head representation of each training split, one per partition leaf for the classifiers, so the
entry carries the label "FinTFM with a trained linear readout". With the `raw` option the readout
also sees the row's own features, squashed on the training rows (§222, §223); it is then a linear
model over the representation and the features, and must be described so. §209 showed that the
readout's advantage over the head on V4FinBench is label count, not a better reading of the
representation, so none of this is evidence that the head is the bottleneck on credit.

| scope | Elo | position | source |
| --- | --- | --- | --- |
| 27 binary datasets, probe plus squash plus partition | 1,000.2 (+156 / -167) | 100th of 107 in its own leaderboard; level with `RF (default)` at 1,000.0, 13 better and 14 worse on paired datasets | §215 |
| 30 binary datasets, `select` columns for the wide tables | 974 (+138 / -169) | | §220 |
| multiclass, 8 datasets | 865 (+120 / -293) | | §224 |
| regression, 13 datasets | 578 (+109 / -297) | | §223 |
| **all 51 datasets, no exclusion** | **903 (+102 / -110)** | **107th of 110** | §224 |

Reference rows in the same leaderboard: `RF (default)` 1,000, `Linear (tuned + ensembled)` 936,
`Linear (tuned)` 919, `KNN (tuned + ensembled)` 967 (§223, §224). The overall entry is below
`RF (default)` and both tuned linear entries. The zero-shot default scores 756 Elo on the 27
binary datasets (§204). The progression of the overall figure was 880 (§221), 888 (§222), 898
(§223), 903 (§224), each step a different multiclass or regression run merged by
`merge_fintfm_lite.py`.

**What the evidence does and does not support.**

- **The 2026-11-09 reading of §215 as "level with the bar" is correct and narrower than it
  sounds.** The 1,010 Elo bar of `docs/roadmap/TABARENA_BAR.md` does not carry over to the leaderboard
  of that run (`KNN (tuned + ensembled)` 999.8, `LR (tuned)` 1,016.0), and against `RF (default)` the
  arm is a tie on paired datasets. Five of its 14 losses to RF are credit or bankruptcy sets
  (§215).
- **The binary figure fell from 1,000 to 974 when the three wide tables entered.** APSFailure,
  Bioresponse and kddcup09_appetency rank 102nd, 103rd and 91st of 108 (§220).
- **§215 reproduces exactly on fintfm 0.9.0 from PyPI**: 27 of 27 shared datasets tied against
  the earlier run (§220).
- **Multiclass gains are real in sign and small in rank.** The readout with the row's features is
  better than §221's on 8 of 8 datasets (p 0.01), and four partition bags move Elo by 40 more, but
  bagging is worse on three datasets and its own sign test is p 0.07 against the §221 control
  (§224). The head's probabilities and a 4,000-row context were null (§224).
- **No part of this has a V4FinBench number.** Squash is null on V4FinBench (§211). The probe per
  partition leaf was harmful there at -0.0570 mean AP on 0 of 5 folds (§218); §219 traced that to
  a sampler defect in the harness (the default-rich leaf got no probe), and the five-fold rerun with
  the repaired sampler, which is the verdict, has not been recorded in FINDINGS. Until it is, a
  TabArena entry built on this readout must not be described as a credit configuration, per
  CLAUDE.md's V4FinBench-first rule. The readout options added in §222-§224 have not been scored
  on binary tasks at all (§223).
- **Single fold.** TabArena-Lite runs fold 0; per-dataset statements are single draws. The Elo
  intervals are about ±100 to ±170.

**Do not write** that FinTFM "beats" or "matches" random forest on TabArena, or quote 903 Elo
without "107th of 110", "fold 0 only" and the readout label. DR7.6 leaves open whether this counts
as near enough to the bar for a submission; that is an owner decision.
