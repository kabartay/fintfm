# Tasks

- [ ] 11.1 Falsification test, no new modelling: score the existing per-horizon UCI panels
      with one checkpoint and check whether cumulative PD is monotone across horizons 1-5 for
      each firm-like row. Verify: report the violation rate in `docs/FINDINGS.md`. A high rate
      is the argument for this change; a near-zero rate weakens it to an efficiency claim.
- [ ] 11.2 Add a monotonicity diagnostic to `evaluation/metrics.py` so term-structure
      incoherence is reported wherever multiple horizons are evaluated. Verify: a test with a
      deliberately inverted term structure that asserts it is flagged.
- [ ] 11.3 Extend `prior/financial.py` to sample a survival process and emit a hazard path
      rather than one Bernoulli label, keeping the macro regime parametric. Verify:
      `uv run pytest tests/test_prior.py -q` plus a test that cumulative hazard is
      non-decreasing by construction.
- [ ] 11.4 Add a multi-horizon head to the model, predicting K hazards per row. Verify: a
      test that cumulative PD is non-decreasing in the horizon for random inputs.
- [ ] 11.5 Compare joint multi-horizon against independent per-horizon models at matched
      compute, on AUC per horizon and on coherence. Verify: numbers into `docs/FINDINGS.md`,
      re-derived from `results.json`, with the paired test from `metrics.py`.
- [ ] 11.6 Write the IFRS 9 framing down properly before any customer conversation: what
      lifetime ECL requires, what the model supplies, and what it does not (LGD, EAD,
      discounting). Verify: a section in `docs/STRATEGY.md` that states the gaps before the
      capabilities.
