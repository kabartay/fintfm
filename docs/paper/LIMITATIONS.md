# Limitations

Written before the results section, not after it. Each item names the finding that establishes
it, so none of this is hedging language — every entry is a measured fact.

## Behind the incumbent on accuracy and calibration

On the V4FinBench out-of-time split, per-horizon logistic regression fitted on all 72,622
training rows leads on mean AUC **0.8616 to 0.8209** and on expected calibration error by
roughly fivefold (~0.0018 against 0.0095). Three of four horizon differences were significant
after Holm correction at the previous best (§32). Claim 6.

The gap closed from 0.142 to 0.041 in one day of inference-time changes with no retraining,
which is the reason to think it is closable — and it is not closed.

**A second, more rigorous measurement (§60/§69/§71, 2026-09-13/14) says the same thing more
precisely and less favourably.** On V4FinBench's own published protocol — five folds, tuned
gradient-boosting baselines, average precision rather than ROC-AUC at this 0.38% base rate
(D13) — fintfm ties logistic regression (Holm p = 0.204) and loses to CatBoost, XGBoost and
LightGBM by 0.14-0.17 AP, all Holm p < 0.001. Reading ROC-AUC with untuned baselines instead
would have placed fintfm second of five; AP with tuned baselines places it third. **Neither
measurement has been re-run on the architecture that closed the synthetic capacity cap**
(§78) — see the new limitation below.

## The comparison to the published benchmark does not exist yet

Every number here uses an out-of-time split of our own design. V4FinBench's published protocol
is 5-fold company-grouped cross-validation, its horizon tasks are built on different rows, its
inference context is 10,000 rows rather than 2,000, and its TabPFN is fine-tuned on the data
(§36). **We cannot say where we would place on their table.** Out-of-time is the harder split,
so the difference does not flatter us, which makes it a precision problem rather than a
credibility one — but it is unresolved either way.

## Context size is capped by pretraining, and larger tasks are priced out

Uniform context peaks at 2,000 rows and *falls* at 4,000, because pretraining used tasks of
256-1,024 rows (§31, §33). The obvious response — pretrain on larger tasks — is measured as
prohibitive on the available hardware: step cost is worse than quadratic in task size, at 32×
for 2,048-row tasks and 500× for 4,096 (`docs/COMPUTE.md`). So a large part of the remaining
gap is a compute limitation rather than a modelling insight, and it will not be argued away.

## Retrieval breaks batch independence, and one firm moved 0.64

Blind context strategies keep a documented property: a prediction depends only on the context
and the query, so chunked scoring is exact. Retrieval makes a firm's context depend on its
query group-mates. At the portfolio level the approximation costs about 0.01 AUC and keeps
roughly six-sevenths of retrieval's gain — but **one firm's cumulative PD moved by 0.64**
depending on which firms were scored alongside it, and the maximum does not shrink as groups
tighten (§37).

A credit decision is made per obligor. This is a product constraint, and the honest resolution
is exact per-query retrieval for individual decisions — 1.05 s per firm, affordable when
scoring one firm — with grouping reserved for portfolio analytics.

## One panel carries the survival claim

Only V4FinBench has firm identifiers and dates, so it is the only dataset that can carry a
per-firm hazard path or a time-based split at all. The UCI Polish and Taiwanese panels have
neither (§7), so every coherence and term-structure result rests on **one panel, one economy
group, four countries, 2006-2021**. The distress label is also a composite financial-distress
criterion rather than a legal bankruptcy filing, which the benchmark's authors state as a
limitation of their own.

## Single seeds in places, and a history of self-correction

§37's grouping numbers, §36's figure readings, and the group-count sweep are single draws.
More importantly, this project produced **five wrong diagnoses in one day** (§28, §30, §32's
monotonicity claim, §34, and the retrieval correction), every one caught by measurement and
none by review (`docs/POSTMORTEM.md`). The corrections all ran in the same direction — each
made the result worse for the project's story before it got better.

That history is a reason to trust the *current* numbers more than the earlier ones, and a
reason to treat any single-draw number here as provisional.

## The prior is unvalidated as a generative model of firms, and its difficulty-matching was actively harmful

It obeys accounting identities by construction. It has never been validated as a realistic
joint distribution of financial statements, and it deliberately samples a macro regime as a
parameter rather than learning real crisis history — because fitting it to real panels would
improve benchmarks while silently destroying the auditability claim, with nothing failing to
warn us (decision D2).

**Correction, 2026-09-14**: this section used to cite §18/§19's "the prior matches real task
difficulty (logistic-regression AUC 0.743 synthetic against 0.769 real)" as a *validation*
point. §42 found the opposite: clamping every synthetic task to a narrow, realistic-looking
difficulty band meant the model never saw a near-deterministic task during pretraining and
never learned to extract sharp signal when one appeared — the training-loop precursor to
§74's much larger capacity-cap finding. Difficulty was widened to span noise-to-near-certain
(§42) specifically because matching real difficulty had been actively teaching the wrong
thing. **Do not cite "the prior matches real difficulty" as a strength in any future writing**;
it is the opposite, on the record, twice.

The financial generator's own label-construction has a separate, independently measured
ceiling: even at a widened sharpness range, it essentially never produces a task with realized
difficulty above 0.99 AUC (0 of 92 sampled tasks), where the generic SCM prior does so 17.5% of
the time (§77) — a structural property of averaging bounded driver weights, not a tuning gap.

## No LGD, no EAD, therefore no ECL

The project produces PD only. Expected credit loss requires PD × LGD × EAD, so this supplies
one of three inputs (`docs/GLOSSARY.md`). Any claim about IFRS 9 provisioning is a claim about
an *input* to provisioning.

## Regulatory terms are used without primary-source verification

Several entries in `docs/GLOSSARY.md` are marked **[verify]** — IFRS 9 stage-transition
mechanics, SICR operational definitions, which Basel version and approach, the specific
supervisory expectations for low-default portfolios, and whether the model's output is
point-in-time or through-the-cycle. **None of these may be asserted to a regulator or in a
paper's framing until checked against a primary source.**

## The architecture fix that closed the synthetic capacity cap — validated on real data 2026-09-15, with one caveat that outlives it

A severe capacity defect — achieved AUC capped at ~0.73 regardless of true task difficulty,
measured against an exactly-known Bayes-optimal AUC — was found (§74), attributed correctly to
architecture rather than prior content after seven content-side hypotheses were eliminated one
at a time (§58-§77), and closed by two-way cell attention in one experiment (§78, regret
0.234-0.277 to 0.001-0.005). **Every number behind that result is synthetic.** No real-data
panel has been scored with the new architecture, and this project has already learned twice
(§47, §74 itself) that a synthetic result and a real-benchmark result can dissociate sharply.
**Task 39.5 has since closed this (§80).** Two complete five-fold V4FinBench runs on the full
1.0M-row panel, differing only in `n_cell_blocks`/`cell_labels`, give the new architecture
+0.0486 mean AP, 5 folds of 5, every Holm-corrected p < 0.001. The synthetic result did not
dissociate.

**What survives is narrower and still limiting**: the fix is established at *matched context*,
and §79's memory cost means the new architecture cannot be run at the context its predecessor
scored best at. Its best reachable score (0.1941) is below the old architecture's best recorded
score (0.2116, §71, single fold). So "the architecture is better" is measured; "the project's
real-data standing is better" is not, and the two must not be conflated.

The reported experiment also deviated from the protocol it was meant to replicate exactly:
`--batch-size 4 --n-rows-choices 256,512` rather than `--batch-size 8 --n-rows-choices
256,512,1024`, forced by a CUDA OOM on the first attempt. **Closed 2026-09-17 (§86)**: the full protocol trains on
the same T4 and passes the step-500 evaluation where the original OOM landed, so the deviation
no longer applies. (§94 corrected the attribution: the fix was §79's hardcoded evaluation batch
size, not feature chunking, which §94 measured as having no effect on training memory.)

## The financial prior's real-data benefit is narrower than the project's own framing assumed

A controlled two-factor design (§61/§63) found the financial generator's structural content
genuinely helps on V4FinBench (+0.098 AP at matched base rate, Holm p = 0.002). The same
comparison on two other real credit panels (Polish and Taiwan bankruptcy, §73/§75) found the
**opposite** ordering on all four measured cells — the generic structural-causal prior, with no
financial content at all, wins by up to +0.065 AP. The benefit measured on V4FinBench does not
generalise to "credit risk" as a domain; it is most plausibly explained by the financial
generator's ratio-construction resembling V4FinBench's own feature-engineering conventions by
construction, which is a narrower and less flattering explanation than domain transfer. Any
claim of domain-specific benefit must name the benchmark it was measured on.

## The architecture that fixed the capacity cap made the best-known inference configuration unaffordable

Cell attention attends across rows within each feature, so its attention cost carries a factor
of the feature count that the pooled architecture does not: `(batch*F, heads, N, N)` with
`N = max_context + query_chunk`. At V4FinBench's 136 features this is ~16 GB at N=2024 and an
estimated ~63 GB at N=4048 (§79).

The practical effect is that **§71's best measured real-data configuration — `max_context=4000`,
`n_ensemble=8`, the one that took single-fold AP from 0.1853 to 0.2116 — cannot be run under
the new architecture on this hardware at all**, and even `max_context=2000` sits past a
measured 92x performance cliff. The validation of §78 therefore runs at `max_context=1000`, a
context §31/§33 measured the *previous* architecture as still improving past. So the two
architectures are compared at a context that handicaps the newer one, and the capacity gain of
§78 and the configuration loss described here have to be weighed together rather than quoted
separately.

This was an engineering limitation rather than a modelling one, and **it is now fixed
(§81)**: chunking row-within-feature attention over `F` is identity-preserving (asserted
byte-for-byte in `tests/test_model.py`) and turned out to be 6.2x smaller *and* 1.6-2.4x
faster, not the memory-for-time trade predicted. `max_context=4000` runs at 22.5 GB, and
chunking is on by default (`feature_chunk: 16`).

**The question is now answered (§84), and this is no longer a limitation.** Both architectures
were re-scored across the context range on the full panel. Each at its own best measured
context, identical rows: cell attention leads by **+0.0417 AP, 5 folds of 5**. The architecture
effect is 6-7x the entire context effect. The concern that the architecture change might be a
net real-data loss is closed, not merely unsupported.

**Corrected 2026-09-16 (§82).** This section previously framed that open question as "whether
cell attention clears the old architecture's best recorded score (0.2116, §71)". That framing
was wrong: 0.2116 is §71's *fold 0*, not its five-fold mean of 0.1676, and §69-§71 ran on a
**10x subsample** (105,900 test rows against §80's 1,000,087), so the two were never measured
on comparable data. The full-panel context effect measured directly is **+0.0069** from
context 1000 to 2000 — an order of magnitude below the -0.066 previously inferred — so the
suggestion that the architecture change might be a net real-data loss is withdrawn.


## Training volume is not the explanation for the accuracy gap, at least not at 5x

The most-cited untested explanation for this project's ~0.22-0.24 AP deficit to tuned gradient
boosting has been training volume: every checkpoint has come from 48,000 synthetic tasks
against a field norm near 10^7, a ~200x shortfall recorded in `CLAUDE.md` since 2026-09-10.

**Tested at 5x and rejected (§93).** A 240,000-task checkpoint differing from its baseline in
nothing but volume scores **-0.0012 AP** across five folds, 3 of 5 nominally positive, with the
only individually-significant fold favouring the *smaller* run.

This does not establish that scale never helps — 5x is a small step on a 200x shortfall, and
it measures the local slope rather than the asymptote. What it does establish is that the cheap
version of the scale argument is dead, and that a large scale run must now be justified by
something other than "we are obviously under-trained."


## The accuracy deficit is not specific to credit, to imbalance, or to the prior's domain

Every real-data number this project had was corporate default data, so the ~0.22 AP deficit to
tuned gradient boosting had never been separated from the kind of data it was measured on.

**§96 separated it.** On 15 public OpenML binary tasks spanning 2.3%-44.5% prevalence and 4-72
features, fintfm is behind the best baseline on **14 of 15**, mean deficit **-0.134**, and wins
outright on one. It reproduces on ecology, speech, software defects, medicine and astronomy.

Three candidate explanations are ruled out by that same experiment. It is not credit-specific,
since it travels across five unrelated domains. It is not purely imbalance, since it persists
at -0.094 between 15% and 44% prevalence. And it is not the prior's domain content: swapping
the entire prior from financial to generic structural-causal moves the mean deficit by
**+0.010**, roughly a tenth of the effect.

The baselines in that suite are untuned, so the honest reading is that the deficit is probably
understated: §69 measured tuning as roughly doubling a booster's lead.


## Placed 93rd of 95 on TabArena, the field's own benchmark

Every accuracy claim in this ledger before 2026-09-20 was measured against baselines this
project ran itself. §98 is the first placement against the field: `fintfm-0.3` registered as a
model in TabArena and run under TabArena's protocol — their preprocessing, their splits, 8
bagged child models per dataset — against a cached leaderboard of 95 methods.

**Rank 93 of 95. Mean ROC-AUC 0.7642 against leaders at ~0.90.** Below every linear baseline;
above only `KNN (default)`. 26 of 27 eligible datasets, zero failures.

Three things bound it, and none of them rescue it. Coverage is **51%** of the suite, since
binary-only excludes 13 regression and 8 multiclass datasets. Categoricals are label-encoded
because fintfm's cell embedding takes numeric scalars only, which makes categorical-heavy
results a floor rather than a fair score. And a CUDA-only device check sent an earlier run to
CPU and produced a spurious timeout, which was an integration defect and not evidence about
the model.

The one encouraging pattern: **two of its three best datasets are the corporate bankruptcy
panels it was designed for** (`taiwanese_bankruptcy_prediction` 0.9291,
`polish_companies_bankruptcy` 0.8436), with `GiveMeSomeCredit` sixth. The model is least bad
where it was aimed.
