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

## The prior is unvalidated as a generative model of firms

It matches real task *difficulty* (logistic-regression AUC 0.743 synthetic against 0.769 real,
§18/§19) and obeys accounting identities by construction. It has never been validated as a
realistic joint distribution of financial statements, and it deliberately samples a macro
regime as a parameter rather than learning real crisis history — because fitting it to real
panels would improve benchmarks while silently destroying the auditability claim, with nothing
failing to warn us (decision D2).

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
