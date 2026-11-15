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
- [ ] 17.5 **Progress (2026-11-15): the Verify is now one runnable command, not yet run. `fintfm-ctxsweep --seeds 0,1,2` already writes the Holm-adjusted paired bootstrap (task 32.3; 1,000 resamples per comparison, whole companies resampled, every pair of strategies per context size, seed and horizon) into the same `context_sweep.json`, so no code was missing; what was missing was the command on record. Command, on the checkpoint §32, §33 and §37 used (`runs/v4-hazard-ldp.pt`, present in the main checkout), on today's embargoed split: `uv run fintfm-ctxsweep --model runs/v4-hazard-ldp.pt --seeds 0,1,2 --out runs/ctxsweep-17-5`. Add `--split legacy` to reproduce §33's leaking split instead. SMOKE-TEST only (two seeds, two strategies, one context size of 200 rows, 6,000-row panel, 100 resamples, machine at load 22): 38 s wall clock including environment setup, which says nothing about the result. ESTIMATED real cost, from §32's measured 24-155 s per cell and not measured for this command: 12 cells per seed, so roughly 45 to 90 minutes for three seeds plus the bootstrap, which is unmeasured at full size; run it on an idle machine and record the wall time.** **Progress (2026-11-15): `fintfm-ctxsweep --seeds 0,1,2` was run and is reported in `FINDINGS` §33 (three seeds per cell, mean +/- std; retrieval beats uniform at every size, and §32's "rises monotonically" was noise). Missing: §33 does not report the paired bootstrap re-run the Verify names, and §33 is flagged as re-measure before quoting (§56, §229).** Three seeds on §32, since a decision now rests on a single draw per cell. Verify:
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
- [ ] 17.7 **Progress (2026-11-15): the Verify is now one runnable command, not yet run. `fintfm-retrgroup` gained `--seeds` (each seed redraws the stratified subsample and re-runs the exact reference) and `--n-boot` (a paired bootstrap of grouped-minus-exact AUC per group size and horizon, whole companies resampled, Holm-adjusted across every seed, size and horizon); it writes the pooled record, with mean and standard deviation of the AUC delta across seeds, to `retrieval_grouping.json` and each seed's own record under `seed<N>/`. Command, on the checkpoint §37 used (`runs/v4-hazard-ldp.pt`, present in the main checkout), on today's embargoed split, with §37's group sizes: `uv run fintfm-retrgroup --model runs/v4-hazard-ldp.pt --n-positives 150 --n-negatives 3000 --group-sizes 2,8,32,128,400 --seeds 0,1,2 --out runs/retrgroup-17-7`. Add `--split legacy` for §37's leaking split. The subsample is 3,150 firms at 4.8% defaulting, which is a lower rate than §37's subsample but still not V4FinBench's 0.380%; say so when quoting it. SMOKE-TEST only (two seeds, 100 firms, two group sizes, 3,609-row pool, 200-row context, 100 resamples, machine at load 22): 38 s wall clock including environment setup, which says nothing about the result. ESTIMATED real cost, from §37's measured 1.05 s per exact query and 306 s for the five grouped runs over 400 queries, scaled linearly to 3,150 queries: about 55 minutes of exact reference plus about 40 minutes of grouped runs per seed, so roughly 5 hours for three seeds before the bootstrap and under whatever load the machine carries; record the wall time.** **Progress (2026-11-15): the concern was answered at a real base rate by 17.8 (`FINDINGS` §70, §71: V4FinBench 0.380%, grouped retrieval drops AP by 0.133), but not with this Verify's command; `fintfm-retrgroup --n-positives 150 --n-negatives 3000` over three seeds is not recorded in FINDINGS.** Repeat §37 at a realistic base rate. §37's subsample is 89% defaulters, because
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
