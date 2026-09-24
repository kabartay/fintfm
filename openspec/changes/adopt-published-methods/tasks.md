# Tasks

- [ ] 36.1 **Validate §42's prior fix.** Run `fintfm-capability` on the fixed-prior checkpoint
      against the old one and the untrained control. Verify: the trained model clears the
      untrained control by a wide margin on `linear` and `conjunction`, recorded as a finding.
      **This gates everything below** — the fix is currently believed rather than shown.
- [x] 36.2 **Ensemble over context draws and feature permutations**, TabPFN's standard
      practice and entirely untried here. Cheap, and usually a reliable small gain. Verify:
      AUC and ECE against a single draw on V4FinBench, with the added inference cost stated,
      three seeds.
      **Done 2026-09-10, and the answer is no (`docs/results/FINDINGS.md` §45).** Context-draw
      ensembling changes nothing (0.6841 to 0.6838), and label-swap averaging drops the model
      to chance because the discrimination it removes *was* the label-slot asymmetry.
      Implementation kept as a diagnostic: a checkpoint whose score survives label-swap
      averaging has signal in the evidence rather than in the labelling. Revisit once one
      does.
- [ ] 36.3 **In-context scaling to larger tables**, following TabICL. §33 measured our model
      degrading past 2,000 context rows, and §31 traced the ceiling to pretraining on
      256-1,024-row tasks. Verify: whether pretraining on larger tasks lifts the ceiling, or
      whether the limit is architectural — `docs/infra/COMPUTE.md` prices 2,048-row tasks at 32× per
      step, so cost the experiment before running it.
- [ ] 36.4 **Report native coverage** — the fraction of datasets or tasks scored without
      falling back — alongside accuracy, everywhere. Seldon running 100% of TabBench while two
      of eight models silently failed on 19% and 29% is a real result, and an average over
      only the easy cases flatters a model exactly as §25 warned. Verify: every experiment
      harness reports a coverage fraction beside its aggregate, and a test asserts that
      attempted tasks equal scored plus explicitly-skipped.
- [ ] 36.5 **Record the fine-tuned-variant decision in `docs/design/DECISIONS.md`**, either way. D2
      forbids training on real data because auditability is the product; Kostrzewa et al. show
      fine-tuning works and transfers across economies. Whether both offerings can coexist is
      a founder's call, and the task is only to write the answer and its reversal condition
      down rather than leave it implicit. Verify: a numbered decision in `docs/design/DECISIONS.md`
      with its alternatives and reversal condition, cross-referenced from D2.
