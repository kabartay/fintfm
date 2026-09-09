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
      impractical. Verify: timings into `docs/COMPUTE.md`, measured on the same hardware as
      §23's context sweep.
      **Done 2026-09-09 (§32).** Roughly 2.5x blind sampling on 47,378 queries: 24/33/61 s
      for uniform against 44/73/155 s for retrieval at 1,000/2,000/4,000 context rows. Usable
      for batch scoring; the batch-independence loss matters more than the time for a
      real-time API.
- [x] 17.3 Compare all four strategies at matched context size on the UCI panels and
      V4FinBench, with the paired bootstrap from `metrics.py`. Verify: numbers into
      `docs/FINDINGS.md`, re-derived from the run.
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
- [ ] 17.6 Measure the grouped-versus-exact retrieval gap on a subsample. The grouped mode is
      an approximation adopted for cost and its error has never been quantified. Verify:
      `retrieval_groups=0` against the grouped default on a few thousand queries, with the
      AUC and ECE difference recorded.
