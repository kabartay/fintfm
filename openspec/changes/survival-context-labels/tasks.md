# Tasks

- [ ] 31.1 Cheapest falsification first: score the existing checkpoint with contexts filtered
      to firms whose outcome is known across the whole grid, versus unfiltered. Verify: if
      horizon-3 AUC moves materially on filtering alone, timing evidence in the context is
      confirmed to matter before any architecture work is done.
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
      `fintfm-v4oot` on both checkpoints, numbers into `docs/FINDINGS.md` with the pre-registered
      prediction quoted and marked hit or miss **before** any interpretation.
- [ ] 31.6 Only if 31.5 succeeds: per-horizon base-rate correction, now that per-horizon
      context rates are recoverable. Verify: ECE at horizons 2-3 against §28's scalar-shift
      numbers (0.0056 and 0.0114), monotonicity still asserted.
