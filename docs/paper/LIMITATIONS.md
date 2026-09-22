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

**No regression head exists**, which is the binding constraint here rather than an oversight:
LGD and EAD are continuous targets, and every checkpoint this project has trained is a
classifier. `regression-and-multiclass` (46.3–46.6) proposes a binned distributional head with
bin edges taken from context quantiles, specifically because LGD is bounded in [0, 1] and
piles up at both ends — a point estimate or a Gaussian head would average the two modes into a
middle value that occurs rarely in the data. **Until that lands, "IFRS 9" should not appear in
a claim without this sentence attached.**

## Multiclass works but is measured once, against a weak ceiling

§99 verified that a `--max-classes 10` checkpoint clears its own untrained control at K = 3, 5
and 10. That is the floor, not the bar. Against multinomial logistic regression — the
correctly-specified model for the probe — it is behind by **0.211 / 0.262 / 0.359** macro
one-vs-rest AUC, a deficit that *widens* with class count. One checkpoint, one probe family,
three seeds: `SINGLE DRAW` in the ledger's vocabulary, and not quotable externally.

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

**§100 then decomposed the rank, and the decomposition matters more than the rank.** Split by
categorical content, the gap to tuned logistic regression is **−0.0320** on the eight datasets
with no categorical columns and **−0.0894** on the nine that are mostly categorical; by
maximum level count it runs −0.0320 / −0.0437 / **−0.1109**, correlating **−0.668** with log
cardinality. **82% of the distance to the method ranked #89 sits on datasets carrying
categorical columns.** So the headline 0.0527 mean gap overstates the modelling problem: the
honest figure is the numeric-only **−0.0320**, and the rest was a preprocessing decision.

That cuts both ways as a limitation. It is good news for the work queue and bad news for the
published number — **§98's rank measured a workaround as much as a model**, which is exactly
why no leaderboard PR has been submitted.

**Fixed, and the rank still did not move (§101).** Out-of-fold target statistics lifted the
mean to 0.7823 and destroyed the cardinality correlation (−0.668 → −0.025), with the eight
numeric-only datasets unchanged to four decimals. Rank: **93 of 95, unchanged.** The residual
is now uniform at ~0.035 across every bucket, so the deficit is one population rather than
two — sharper to attack, and no smaller where it counts. **The project's stated target,
untuned trees < fintfm < tuned trees, is not reached:** default random forest leads on 21 of
27 datasets.

The one encouraging pattern: **two of its three best datasets are the corporate bankruptcy
panels it was designed for** (`taiwanese_bankruptcy_prediction` 0.9291,
`polish_companies_bankruptcy` 0.8436), with `GiveMeSomeCredit` sixth. The model is least bad
where it was aimed.

## Scale cannot close the gap, and a peer's published curve is why we know

§104 closed `column_id_dim`, §102 found no axis in the uniform ~0.035 residual and §103
weakened the mechanism it proposed, leaving **scale** as the only standing explanation. §108's
first attempt at it lost 0.0077 on TabArena binary — on a confounded design, since the larger
model ran at batch 4 against the baseline's batch 8 and therefore saw **half the tasks** at
matched steps. A matched-task rerun is in flight and a confounded negative is not a negative.

**But the hypothesis was already capped by evidence we did not generate.** Synthefy's Nori
publishes a clean scaling curve for a synthetic-prior in-context model: 6M → 100M parameters,
a 16.7× increase, returns **+0.0049 R² on TabArena** and **+0.0034 overall across 95 tasks**.
The metric is not ours and the number does not transfer. The order of magnitude does: in this
model class an order-of-magnitude scale-up returns single-digit thousandths, and our deficit is
**0.035**.

**Parameters and data are separate axes, both tested here, and the confound is now removed.**
§93 scaled *tasks* 5× (48,000 → 240,000) and measured **−0.0012 AP** across five folds. §108
scaled *parameters* 5.7× and measured −0.0077 on TabArena binary, but at half the baseline's
task count. §114 reran it at **matched tasks** and measured **−0.0049** (12/27, p = 0.701): the
confound explained +0.0028 of the loss and not the rest. Nori scales parameters 16.7× for
+0.0049 R². And the deep-narrow arm — the other corner of the design space, and the one §108's
width-only change had left untested — **could not be scored at all**, raising
`TimeLimitExceeded` after 8 of 27 datasets, because depth costs inference time and this
project's predict time is already 8.6 s/1K against a field norm near 0.1.

Four measurements, no gain anywhere near 0.035, and one configuration that cannot be ranked
regardless of what its accuracy would have been.

**TabDPT is the apparent counter-example, and reading it carefully makes the picture worse
rather than better.** Ma, Thomas et al. (arXiv:2410.18164) report that scaling both model and
data "leads to consistent performance improvements that follow power laws". But their
contribution is precisely that **real** data beats synthetic: "incorporating real data during
the pre-training phase can lead to significantly faster training and better downstream
generalization", against "purely open-source synthetic data generators". Their power laws are
evidence for *real-data* scaling. They are not evidence that synthetic-prior scaling follows
power laws — and §93, §108 and Nori's curve are three independent observations that it does
not.

**Three groups argue synthetic-only saturates, and the leaderboard says otherwise.** §110
classifies all 95 TabArena methods by pretraining corpus: **every one of the top 14 ranks is
synthetic-pretrained**. LimiX-2 is first at Elo 1872 on SCM-generated synthetic data alone;
TabPFN-3.5, Mitra-v2 and TabICLv2 follow. The first entry with any real data is RealTabPFN-2.5
at 15 — itself synthetic-pretrained with real adaptation — and the best model trained on a real
corpus is TabDPT-Turbo at **21**. TabDPT, ConTextTab and iLTM are each right about their own
ablations and none of them places above the synthetic frontier.

**So decision D2's synthetic-only constraint costs nothing measurable in rank.** It forecloses
an axis three groups are actively mining, and the benchmark says that axis is not where the
frontier is. **MITRA** (arXiv:2510.21204) shows what does move: synthetic-only, beating TabPFNv2
and TabICL, at **rank 8** — with **prior design** as its entire contribution, a curated mixture
chosen for *performance, diversity and distinctiveness*.

**That relocates the limitation rather than removing it.** The evidence does not say synthetic
data saturates; it says synthetic data saturates **when the prior is wrong**, and that a better
prior beats a bigger corpus. Which is the best news this project's thesis has had — a financial
prior is exactly a bet on prior design — and simultaneously the sharpest indictment of where its
effort has gone: §54, §104 and §44 are architecture, while MITRA reports its priors are
**model-agnostic**, improving both 1D row attention and 2D element attention.

**The specific, named gap is a tree-based prior.** MITRA finds that TFMs pretrained on SCMs
"do not always generalize well to all types of data generated from TBPs" — gradient boosting,
random forest, decision tree, extra tree. This project's mixture is financial (0.7) and SCM
(0.3) with no tree prior at all, while every baseline it loses to is a tree ensemble and real
tabular data is full of threshold structure. A draft must state that the prior mixture was never
evaluated against MITRA's three criteria, because that is a gap in the work rather than a
property of the domain.

**Decision D2's cost is therefore smaller than two earlier drafts of this section claimed,
and the drafting itself is the limitation worth reporting.** This entry was rewritten three
times in one session — once per arriving abstract — before anyone checked the rank ordering that
settles it (§110). A related-work argument assembled in the order papers arrive is a reading
log. What remains a genuine cost is narrow: real adaptation *on top of* synthetic pretraining
(RealTabPFN-2.5, rank 15) is a route D2 forecloses and the frontier does use, so the honest
statement is that D2 costs the adaptation step, not the foundation.

**State this, because "the model is small" is the explanation a reader will reach for.** It is
the explanation this project reached for too, for weeks, and it is quantitatively insufficient
by roughly an order of magnitude. `openspec/changes/learn-from-peers` task 48.1 tests the
cheapest remaining candidate — that §108 scaled the wrong *dimension*, since Nori-6M is 16
layers at width 128 where our scale-up went wide rather than deep.

The limitation is broader than the finding. **This project spent GPU budget on a hypothesis
that a competitor's public README had already bounded**, and the cost of reading the field was
one afternoon. That is now a standing instruction in `CLAUDE.md`.

## A diagnostic's baseline is the measurement, and ours was wrong for four weeks' worth of reasons

`fintfm-priorscore` scores a synthetic prior on MITRA's three criteria — performance,
diversity, distinctiveness — from fitted baselines rather than from this project's own model.
That design is right and deliberate: a prior diagnostic routed through our checkpoint cannot
separate "the prior does not contain this structure" from "our model cannot learn this
structure", which is the confound §49 and §51 cost days to untangle on the capacity question.

**Its first version was still wrong, in a way that survived review.** The linear arm was a bare
`LogisticRegression` on raw features. Financial ratios are pathologically heavy-tailed — §35
measured 110 of 136 V4FinBench features with a standard deviation over ten times their
interquartile range — so the fit failed to converge on eight of twenty-five tasks. Since
distinctiveness is `tree_auc − linear_auc`, an understated linear arm **inflates** it, worst
exactly where the tails are heaviest.

The consequence was not a small numerical shift. The financial prior moved from **+0.0754 to
−0.0292**, a sign flip and a 0.105 swing (§112). The wrong version said the financial prior was
the *most* tree-favourable member of the mixture — which would have made the tree prior
redundant and was the strongest available argument against building it at all.

**Two things make this worth reporting as a limitation rather than a fixed bug.**

First, the defect was one this project had already found, measured and fixed *elsewhere*. §35
is among the older findings here; `feature_transform="rank"` has been the inference default
since, on the strength of +0.086 AUC across six of six configurations. A finding does not
propagate to new code by having been published.

Second, and more general: **a weak baseline behaves differently in a diagnostic than in a
benchmark.** In a benchmark it flatters the model under test and a reader notices. In a
diagnostic it silently *becomes* the measurement, and the resulting number need not look
suspicious — +0.0754 for a prior full of accounting thresholds and policy cutoffs is exactly
what one would expect to see.

The rule now in force: a fitted baseline used as an instrument gets the same preprocessing the
model under study gets, and a convergence warning inside a measurement is a failed measurement
rather than a log line. Any prior-selection statistic in a draft must state its baseline's
preprocessing explicitly, because the number is meaningless without it.

