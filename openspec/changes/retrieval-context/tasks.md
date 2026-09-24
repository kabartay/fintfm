# Tasks

- [x] 17.1 Implement a `retrieval` context strategy selecting nearest neighbours on
      normalised features, balanced within class. Verify: a test that a query's context is
      closer to it on average than a uniform sample is.
      **Done 2026-09-09**, in `fintfm/inference/retrieval.py` with eleven tests in
      `tests/test_retrieval.py`. **Deviation from the proposal:** the context is *not*
      balanced within class, because §29 measured balancing as costing 10-12 AUC points here.
      A `retrieval_min_positive` floor insures against a group drawing zero defaults, which
      is the failure the proposal's "balance within class" was actually guarding against.
      **Cost noted:** retrieval makes a query's context depend on its group-mates, so
      batch-independence — exact for every blind strategy — no longer holds. Both an exact
      per-query mode and a grouped mode exist so the approximation is measurable; the trade
      is asserted in both directions in `tests/test_retrieval.py`.
- [x] 17.2 Measure inference cost against blind sampling; retrieval must not make scoring
      impractical. Verify: timings into `docs/infra/COMPUTE.md`, measured on the same hardware as
      §23's context sweep.
      **Done 2026-09-09 (§32).** Roughly 2.5x blind sampling on 47,378 queries: 24/33/61 s
      for uniform against 44/73/155 s for retrieval at 1,000/2,000/4,000 context rows. Usable
      for batch scoring; the batch-independence loss matters more than the time for a
      real-time API.
- [x] 17.3 Compare all four strategies at matched context size on the UCI panels and
      V4FinBench, with the paired bootstrap from `metrics.py`. Verify: numbers into
      `docs/results/FINDINGS.md`, re-derived from the run.
      **Done for V4FinBench 2026-09-09 (§32):** retrieval beats uniform by +0.0665/+0.0632/
      +0.0952 at the first three horizons, all Holm-significant; the fourth is inconclusive
      on 18 positives. **Still open for the UCI panels**, which have no firm identifiers and
      so cannot carry the survival protocol (§7) — the binary-path comparison belongs to
      `revisit-context-strategy` task 32.2 and is tracked there.
- [x] 17.4 If retrieval loses, record it plainly and state what it implies: context quality
      is not the lever, so §23's flatness points at model capacity and effort should move to
      scale. Verify: a finding that says so without hedging.
      **Not triggered: retrieval won (§32).** Context quality is the lever. Recorded here
      because the contingency was pre-registered and resolving it explicitly is the point.
- [ ] 17.5 Three seeds on §32, since a decision now rests on a single draw per cell. Verify:
      `uv run fintfm-ctxsweep --seeds 0,1,2` with the paired bootstrap re-run.
- [x] 17.6 Measure the grouped-versus-exact retrieval gap on a subsample. The grouped mode is
      an approximation adopted for cost and its error has never been quantified. Verify:
      `retrieval_groups=0` against the grouped default on a few thousand queries, with the
      AUC and ECE difference recorded.
      **Done 2026-09-09 (`docs/results/FINDINGS.md` §37), via `fintfm-retrgroup`.** About 0.012 AUC
      at the portfolio level, mean absolute deviation 0.0004-0.0009 in cumulative PD, Spearman
      0.98-0.997 — so grouping keeps roughly six-sevenths of retrieval's gain. Exact retrieval
      costs 1.05 s/query, measured, against 0.0023 s/query grouped: ~450x, confirming the
      "~500x" the module docstring had been asserting unmeasured.
      **But one firm moved 0.238 in cumulative PD**, which makes batch dependence a product
      constraint rather than an engineering detail: portfolio analytics can group, an
      individual credit decision should use exact retrieval, and 1.05 s for one firm is
      affordable in that context.
- [ ] 17.7 Repeat §37 at a realistic base rate. §37's subsample is 89% defaulters, because
      every positive was kept to keep the AUC column non-degenerate while the exact reference
      stayed inside an hour. The deviation and Spearman columns do not depend on labels and
      stand; the **AUC deltas do not transfer to a real book**. Verify: `fintfm-retrgroup
      --n-positives 150 --n-negatives 3000`, three seeds, with the AUC delta re-derived.
- [x] 17.8 **Done 2026-09-13/14 (`docs/results/FINDINGS.md` §70, §71).** Task 17.7's concern was
      confirmed and sharpened: at V4FinBench's real regime (0.380% base rate, one fold, six
      configurations, reproduced twice to four decimals), grouped retrieval alone drops AP by
      **0.133** relative to uniform sampling — not merely "deltas don't transfer," actively
      harmful. Not rescued by widening context or ensembling; every combination including
      retrieval scores below the uniform baseline. Dropping retrieval and keeping
      `n_ensemble=8` + wider `max_context` instead reached the best AP measured for this
      checkpoint (+0.031 AP over the old defaults, five-fold paired bootstrap, Holm p<0.001,
      §71). **The mechanism is still open** (task 38.13 in `cross-domain-coverage`): a
      candidate — grouped retrieval targets the *centroid* of up to ~330 averaged queries in
      130-d space, which may not resemble any individual query closely enough to retrieve a
      useful context — was proposed and a test was set up (exact per-query retrieval,
      `retrieval_groups=0`, on a small enriched subsample) but the run was killed mid-flight
      during a machine-load safety stop (2026-09-14) before it produced a result. **No
      exact-retrieval number exists yet; the centroid-averaging hypothesis is untested, not
      ruled out.** Re-running it is the natural next step for task 38.13.
      **Until the mechanism is understood and fixed, retrieval should not be the default
      context strategy for scoring at a low base rate**, regardless of §32/§37's earlier
      positive measurements at other regimes.
