# Tasks

- [x] 11.1 Falsification test, no new modelling: score the existing per-horizon UCI panels
      with one checkpoint and check whether cumulative PD is monotone across horizons 1-5 for
      each firm-like row. Verify: report the violation rate in `docs/FINDINGS.md`. A high rate
      is the argument for this change; a near-zero rate weakens it to an efficiency claim.
      **Done 2026-09-08 — the change is FOUNDED. 11.0% violation rate per step; only 60.6%
      of firms get a fully monotone curve, while the portfolio aggregate is monotone and
      hides it entirely. `docs/FINDINGS.md` §11.**
- [x] 11.2 Add a monotonicity diagnostic so term-structure incoherence is reported wherever
      multiple horizons are evaluated. Verify: a test with a deliberately inverted term
      structure that asserts it is flagged. **Done 2026-09-08:
      `hazard.coherence_violations`, and `test_coherence_violations_detects_a_broken_curve`.**
- [ ] 11.3 Extend `prior/financial.py` to sample a survival process and emit a hazard path
      rather than one Bernoulli label, keeping the macro regime parametric. Verify:
      `uv run pytest tests/test_prior.py -q` plus a test that cumulative hazard is
      non-decreasing by construction.
- [x] 11.4 Add a multi-horizon head to the model, predicting K hazards per row. Verify: a
      test that cumulative PD is non-decreasing in the horizon for random inputs.
      **Done 2026-09-08. `HazardHead` + `FinancialTFM.term_structure`, 12 tests. Monotone by
      construction, including under weights scaled 500x. 0 violations in 12,000 steps
      against 11.0% for the per-horizon approach (`FINDINGS` §20).**
- [x] 11.5 Compare joint multi-horizon against independent per-horizon models at matched
      compute, on AUC per horizon and on coherence. Verify: numbers into `docs/FINDINGS.md`,
      re-derived from `results.json`, with the paired test from `metrics.py`.
      **Done 2026-09-08, `FINDINGS` §21, `fintfm-termstruct`. Discrimination is a tie
      (+0.0039 at matched total, -0.0032 when the baseline gets 5x compute), coherence goes
      from 12-29% violations to 0%. Coherence is free. Paired test still owed - see 11.7.**
- [ ] 11.7 Repeat §21 across >= 3 seeds with the paired bootstrap, since the AUC differences
      (+/-0.004) are almost certainly within noise and "no difference" is currently an
      inference rather than a measurement. Verify: mean +/- std per arm and a Holm-corrected
      verdict in `docs/FINDINGS.md` §21.
- [ ] 11.6 Write the IFRS 9 framing down properly before any customer conversation: what
      lifetime ECL requires, what the model supplies, and what it does not (LGD, EAD,
      discounting). Verify: a section in `docs/STRATEGY.md` that states the gaps before the
      capabilities.
