# Tasks

- [x] 34.1 **Done (2026-11-15): the full suite is green on CI at 2eb8927 (which carries the 34.1 code from b376341 and the recipe pins), meeting the Verify's `uv run pytest -q` clause; the targeted tests in tests/test_joint_objective.py pin the default loss and the weighted sum.** **Progress (2026-11-15): implemented as `train.joint_classification_weight` (`--joint-classification-weight`, default 0.0 = survival loss alone, unchanged); a batch with `period` then trains `survival + weight * classification` and records both in `trained_objectives`, the weight in the checkpoint's `train_config` provenance; refused without a hazard head, with a non-binned head, with `--train-only`, or when negative. `tests/test_joint_objective.py` (5 tests) shows a short CPU run passing `assert_trained_for` for both paths. Left open: the full `uv run pytest -q` half of Verify was not run, as the machine was busy; close once it is green.** Sum the objectives with a configurable weight when `period` is present, recording
      both in `trained_objectives`. Verify: a short run producing a checkpoint that passes
      `assert_trained_for` for both paths, and `uv run pytest -q` green.
- [ ] 34.2 Pretrain at matched compute against `runs/v4-hazard-ldp.pt` and compare on the
      V4FinBench out-of-time split. Verify: mean AUC and ECE against §35's 0.8118 / 0.0126,
      with the weight recorded.
- [ ] 34.3 Score the same checkpoint on the binary path across Polish and Taiwan, against the
      classification-only checkpoint. Verify: whether one checkpoint can match two, stated
      either way.
- [ ] 34.4 If joint training wins, retire the two-checkpoint workflow and say so in
      `docs/design/DECISIONS.md`; if it loses, record that the objectives conflict and keep the
      guard as the fix. Verify: a decision with its reversal condition.
