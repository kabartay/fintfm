# Tasks

- [ ] 41.1 **Wait on `cell-attention-and-task-inference` task 39.7's result.** **Progress (2026-11-15): the 39.7 probe harness (`fintfm.experiments.task_probe`) is built; its measured result is still outstanding.** Verify: this
      proposal's first pretraining task (41.3) cites the probe-separation finding that
      justifies starting it, not merely a calendar date.
- [x] 41.2 **Done (2026-11-15): `ModelConfig.task_token` (default False) adds a learned query that cross-attends over the context rows' final row-stack states and is added to every row through a zero-initialised projection before the final norm and head; `predict_context(stage="task_token")` exposes `z_task` for `task_probe` (via `transform_representation`, accepted at run time, not in the golden-pinned annotation). Default outputs are byte-identical (golden tests unchanged and green; an old state_dict and a checkpoint without the key load strictly with identical outputs); tests cover the 3-step CPU train with the eval path, context-row-order invariance, no query-to-query leakage and the parameter count. Unmeasured: 41.3 measures whether it helps.** **Design the explicit task token** as a config-gated addition
      **Prior art, found 2026-09-21: TabSwift's register tokens** (`github.com/LAMDA-Tabular/
      TabSwift`) are a published implementation — learnable tokens prepended to the ICL
      sequence, carrying dataset-level information "without interfering with the data tokens",
      discarded after the final layer so only data positions decode. Design against that rather
      than from scratch, and note it is a *dataset*-level slot, orthogonal to `column_id_dim`
      (which carries column identity and which §104 closed at 16).
      (`ModelConfig.task_token: bool = False`, matching the established additive-change pattern
      of `pooling`, `column_id_dim`, `n_cell_blocks`), pooling across context rows into one
      `z_task` that query predictions condition on. Verify: `task_token=False` reproduces every
      existing test byte-for-byte, matching how `identity_shuffle=False` and `n_cell_blocks=0`
      were regression-tested.
- [ ] 41.3 **Progress (2026-11-15): the matched Kaggle pair is prepared, not pushed and not run.** Kernels `scripts/kaggle/ts_4130_token_s0` (`fintfm-ts-4130-token-s0`) and `scripts/kaggle/ts_4130_ctrl_s0` (`fintfm-ts-4130-ctrl-s0`), written by `scripts/kaggle/train_score/make_train_score.py 4130-token-s0 4130-ctrl-s0`. Both train `configs/published/v4-cellattn-labels.yaml` at commit d58141e (`--task-token` is in it) with fp16, split attention, DDP on a T4x2, per-GPU batch 4, seed 0, 6,000 steps = 48,000 tasks (the §171 recipe), `--checkpoint-every 1000`; the flags differ by `--task-token` alone (`tests/test_4130_pair.py`). Each then scores the checkpoint zero-shot on V4FinBench's five full-panel folds (`SCORE_V4` True, the protocol of every other `ts_*` kernel), so the two are compared against each other on average precision first. **Cost, ESTIMATED:** training 6,000 steps at 0.59 s/step (§233's 35,300 s over 60,000 steps, T4x2, 8 tasks per step) is about 3,500 s, about 1 h; the task token's cross-attention adds an unmeasured amount, so the first 15 minutes of step rate are the check. With installation and five-fold scoring, §178's range of 1.5 to 2.5 h per arm gives about 3 to 5 h of Kaggle GPU quota for the pair. **Launch** (two lanes, scoring on Kaggle so nothing is scored locally): `nohup scripts/kaggle/launch_queue.sh lane1 ts_4130_token_s0:- > /dev/null 2>&1 &` and `nohup scripts/kaggle/launch_queue.sh lane2 ts_4130_ctrl_s0:- > /dev/null 2>&1 &`. **Afterwards**, on the fetched checkpoints (a few minutes on CPU, 24 families x 20 tasks): `uv run python -m fintfm.experiments.task_probe --models token=runs/kaggle-fintfm-ts-4130-token-s0/4130-token-s0/4130-token-s0.pt,ctrl=runs/kaggle-fintfm-ts-4130-ctrl-s0/4130-ctrl-s0/4130-ctrl-s0.pt --out runs/task_probe_4130` (path as the lane fetches it; check it), and once more with `--stage task_token` on the token arm alone; the §74 Bayes-ceiling probe on both is still to be run. Open until those numbers exist. **Pretrain with and without the task token**, same protocol as every other T4
      comparison this session. Verify: task 39.7's DGP-classification probe run on both,
      reporting whether the explicit token measurably improves task-family separability, and
      the §74 Bayes-ceiling probe run on both to check it does not regress raw discrimination.
- [ ] 41.4 **Mixture-of-priors routing**, only if 41.3 shows the task token carries real
      information. Verify: `p(k|D)` is inspectable per prediction, and a held-out mixture of
      known families is scored with the routing weights reported alongside the prediction.
- [ ] 41.5 **Posterior task uncertainty on ambiguous contexts.**
      **Progress (2026-11-15): the ambiguous-context construction and Bayes reference for
      the predictive-entropy side exist as cell-attention-and-task-inference 39.19
      (`fintfm.experiments.task_posterior`); the `p(k|D)` entropy side still needs 41.4.**
      Verify: constructed contexts
      consistent with two task families are scored, and `p(k|D)`'s entropy is compared against
      unambiguous contexts of matched difficulty — it should be higher on the ambiguous ones.
- [ ] 41.6 **Retrieval over mechanisms**, reframing `retrieval-context`'s existing work. Verify:
      only started after 41.4 exists to retrieve into; the finding states explicitly whether
      this out-performs `retrieval-context`'s row-level retrieval or merely duplicates it.
