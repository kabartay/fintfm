# Tasks

- [x] 31.1 Cheapest falsification first: score the existing checkpoint with contexts filtered
      to firms whose outcome is known across the whole grid, versus unfiltered. Verify: if
      horizon-3 AUC moves materially on filtering alone, timing evidence in the context is
      confirmed to matter before any architecture work is done.
      **Done 2026-09-09: confounded, not supportive (`docs/results/FINDINGS.md` §31).** The filter
      moved mean AUC +0.006 but the gain was largest at h0 (+0.0102) and smallest at h3
      (+0.0018) — the opposite of the prediction. `n_observed` is censored by default itself,
      so the filter removes defaulters and drops the context rate from 1.54% to 0.19%; the
      gain is §29's base-rate effect, not an observation-depth effect. No cheap unconfounded
      test exists. Tasks 31.2+ now sit behind `retrieval-context`.
- [ ] 31.2 Extend `FinancialTFMClassifier.fit` to accept `period` and `n_observed`, carried
      through context selection intact. Verify: a test asserting the selected context's period
      distribution matches the selected rows, and that omitting them leaves the binary path
      byte-identical.
- [ ] 31.3 Embed the period label in `FinancialTFM.term_structure` over a `K + 1` state grid.
      Verify: `uv run pytest tests/test_hazard.py -q`, plus a test that permuting context rows
      leaves predictions unchanged (the equivariance guarantee must survive the new embedding).
- [ ] 31.4 Emit `period` into the pretraining context, not only the query targets. Verify: a
      test that a task's context period labels agree with its binary labels.
- [ ] 31.5 Pretrain the matched pair and run the pre-registered comparison. Verify:
      `fintfm-v4oot` on both checkpoints, numbers into `docs/results/FINDINGS.md` with the pre-registered
      prediction quoted and marked hit or miss **before** any interpretation.
- [ ] 31.6 Only if 31.5 succeeds: per-horizon base-rate correction, now that per-horizon
      context rates are recoverable. Verify: ECE at horizons 2-3 against §28's scalar-shift
      numbers (0.0056 and 0.0114), monotonicity still asserted.
