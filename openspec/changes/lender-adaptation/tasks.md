# Tasks

- [ ] 16.1 Repeat §23's context sweep on a **corporate** panel (V4FinBench, UCI). Verify:
      numbers into `docs/FINDINGS.md`. **If AUC is flat there too, this whole change is
      solving a non-problem and should be closed rather than built.**
- [ ] 16.2 Implement a low-rank adapter over the frozen base model, training only the adapter
      parameters. Verify: a test asserting base weights are unchanged after fitting.
- [ ] 16.3 Compare adapter against in-context conditioning at matched data on a held-out
      lender-like slice (e.g. one country in V4FinBench). Verify: paired test with the
      bootstrap from `metrics.py`.
- [ ] 16.4 Measure adaptation cost in wall-clock and dollars on a single GPU, since the whole
      argument is economic. Verify: figures in `docs/COMPUTE.md`, measured not estimated.
- [ ] 16.5 Record the provenance rule in `openspec/specs/pretraining-provenance`: adapters are
      per-lender, never merged into a shipped base model. Verify:
      `uv run python openspec/tools/validate.py` passes.
