# Tasks

- [ ] 16.1 Repeat §23's context sweep on a **corporate** panel (V4FinBench, UCI). Verify:
      numbers into `docs/results/FINDINGS.md`. **If AUC is flat there too, this whole change is
      solving a non-problem and should be closed rather than built.**
- [ ] 16.2 Implement a low-rank adapter over the frozen base model, training only the adapter
      parameters. Verify: a test asserting base weights are unchanged after fitting.
- [ ] 16.3 Compare adapter against in-context conditioning at matched data on a held-out
      lender-like slice (e.g. one country in V4FinBench). Verify: paired test with the
      bootstrap from `metrics.py`.
- [ ] 16.4 Measure adaptation cost in wall-clock and dollars on a single GPU, since the whole
      argument is economic. Verify: figures in `docs/infra/COMPUTE.md`, measured not estimated.
- [x] 16.5 **Done (2026-11-15): recorded as P5 in `openspec/specs/pretraining-provenance/spec.md`, with a scenario: adapters and fine-tuned weights trained on a lender's or any real data are a separate artefact that inherits the FinTFM-R side of the provenance table, never the synthetic-only claim; the base checkpoint's claim is unchanged; adapters are per-lender and never merged into a shipped base model. Gap, not implemented here: no code records real-data fine-tuning in a checkpoint. `_training_provenance` (`modeling/train.py`) records `source_shares` (so `p_real_edgar` shows real EDGAR pretraining) and `init_from` (path, SHA-256, tasks seen) for continued pretraining, and `FinancialTFMClassifier(finetune_steps=N)` fine-tunes a copy in memory with no saved provenance. P5 is therefore enforced by review and the 16.2 frozen-base test, not by a checkpoint field.** Record the provenance rule in `openspec/specs/pretraining-provenance`: adapters are
      per-lender, never merged into a shipped base model. Verify:
      `uv run python openspec/tools/validate.py` passes.
- [ ] 16.6 **Score §164's pre-registered fine-tuning arms and write the verdicts into §164.**
      Full fine-tuning of a copy (`finetune_steps`, 0.7.0, D18) is the measured stand-in for
      this change's adapter: §163 raised mean AP 0.1986 → 0.2062 on five full-panel V4FinBench
      folds at one fine-tuning seed. §164 pre-registered three arms from §163's setting: `S1`
      (`finetune_seed: 1`, CPU), `EN` (`finetune_query_pos_frac: 0.1`) and `L600` (600 steps),
      the last two on a Kaggle GPU (`scripts/kaggle/task_164_en`, `task_164_l600`), plus the
      re-run of §163's own setting on the same GPU kernel that §164's 2026-10-15 amendment
      requires before a GPU arm's verdict is more than provisional. Verify: §164 reports the
      seed gap (per fold and mean absolute difference, `S1` against §163's arm), whether §163
      replicates under its own rule, and a verdict for `EN` and `L600` under §164's rules, each
      GPU verdict either beside the measured device gap or labelled provisional
      (device-confounded); the predictions stay unedited.
