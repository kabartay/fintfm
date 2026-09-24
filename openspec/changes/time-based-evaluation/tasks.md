# Tasks

- [x] 3.1 Determine whether the UCI ARFF files carry any per-observation date or period
      field. Verify: print the attribute list from each file. If absent, mark this change
      blocked on `second-credit-panel` and say so in `docs/roadmap/NEXT.md`.
      **Done 2026-09-08: NO dates, no periods, no company identifiers — 64 anonymous
      numeric attributes plus a binary class. This change is BLOCKED on
      `second-credit-panel`; it cannot be done on the only dataset we hold. See
      `docs/results/FINDINGS.md` §7.**
- [ ] 3.2 Implement the splitter in `evaluation/`, refusing to run if any identifier appears
      in both windows. Verify: a test constructing an overlapping panel and asserting it
      raises.
- [ ] 3.3 Report random and time-based numbers side by side in `bench.py`. Verify:
      `uv run fintfm-bench --credit` prints both, and the gap is recorded in
      `docs/results/FINDINGS.md` re-derived from the run output.
- [ ] 3.4 Update `docs/roadmap/STRATEGY.md` Phase 1 to require the time-split number, since the phase
      currently says "with time-based splits" while the harness does not do it. Verify:
      `grep -n "time-based" docs/roadmap/STRATEGY.md` shows the requirement, and the harness refuses to
      run a time split on a dataset with `has_period_labels=False`.
