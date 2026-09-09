# Tasks

- [ ] 34.1 Sum the objectives with a configurable weight when `period` is present, recording
      both in `trained_objectives`. Verify: a short run producing a checkpoint that passes
      `assert_trained_for` for both paths, and `uv run pytest -q` green.
- [ ] 34.2 Pretrain at matched compute against `runs/v4-hazard-ldp.pt` and compare on the
      V4FinBench out-of-time split. Verify: mean AUC and ECE against §35's 0.8118 / 0.0126,
      with the weight recorded.
- [ ] 34.3 Score the same checkpoint on the binary path across Polish and Taiwan, against the
      classification-only checkpoint. Verify: whether one checkpoint can match two, stated
      either way.
- [ ] 34.4 If joint training wins, retire the two-checkpoint workflow and say so in
      `docs/DECISIONS.md`; if it loses, record that the objectives conflict and keep the
      guard as the fix. Verify: a decision with its reversal condition.
