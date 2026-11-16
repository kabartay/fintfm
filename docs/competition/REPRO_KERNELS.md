# Reproduction kernels for the TabArena-Lite entry

Two Kaggle kernels rerun the 51 TabArena-Lite datasets (30 binary, 8 multiclass, 13
regression; repeat 0, fold 0) from the fork's pull-request branch the way a maintainer would:
no `FINTFM_*` variable, no local checkpoint, `fintfm` from PyPI, checkpoints fetched by the
fork's `fintfm/checkpoints.py` from Hugging Face. Each fits the registry entry
`FinTFM-LinProbe`, default config, through `TabArenaV0pt1ExperimentBundle` and
`TabArenaContext.build_and_run_jobs(subset="lite")`, one process per dataset.

| kernel | runs (result names) | local CPU time for comparison |
| --- | --- | --- |
| `task_lite_repro_pr_bin` | `fintfm_pr_repro_bin` (30 binary) | about 6.7 h |
| `task_lite_repro_pr_mcreg` | `fintfm_pr_repro_mc` (8), `fintfm_pr_repro_reg` (13) | about 1.1 h + 4.5 h |

Kaggle time is not measured; the readout is CPU-bound, so the T4 is not assumed faster. Each
kernel stops starting datasets after 11 h.

Both scripts are generated: edit `scripts/kaggle/repro_pr_kernel_template.py` or
`make_repro_pr_kernels.py`, run `.venv/bin/python scripts/kaggle/make_repro_pr_kernels.py`
(`tests/test_repro_pr_kernels.py` fails when they are out of date).

## Preflight (the kernel exits non-zero and says why)

1. `FORK_COMMIT` is not a 7-40 digit hex SHA (shipped as `<FORK_COMMIT>`).
2. `fintfm==FINTFM_VERSION` cannot be downloaded from PyPI (0.11.0 is unreleased).
3. After cloning the fork at `FORK_COMMIT`: `checkpoints.py` still has an `"unpublished"`
   reference, or no reference, for a problem type the kernel runs. The binary kernel needs only
   binary; the mcreg kernel needs multiclass and regression.

## Steps, after the weights are published and 0.11.0 is released

1. Publish multiclass and regression weights (`scripts/publish_weights.py`), and release fintfm
   0.11.0 (see the Releases section of `CLAUDE.md`).
2. In the fork, replace the two placeholder references in `checkpoints.py` with published ones
   (repo, filename, commit, SHA-256) and update the PR branch. Note its head SHA.
3. In both kernel scripts set `FORK_COMMIT` (or edit `make_repro_pr_kernels.py`'s template and
   regenerate). Check `FINTFM_VERSION` equals the release.
4. Dry-run the preflight anywhere with internet (clones the fork, fits nothing):
   `.venv/bin/python scripts/kaggle/task_lite_repro_pr_bin/task_lite_repro_pr_bin.py --preflight-only`
   and the same for `..._mcreg`. Both must print `preflight ok`.
5. Launch with the lane launcher, two lanes, one kernel each, from the repository root:
   `nohup scripts/kaggle/launch_queue.sh lane1 task_lite_repro_pr_bin:- > /dev/null 2>&1 &`
   `nohup scripts/kaggle/launch_queue.sh lane2 task_lite_repro_pr_mcreg:- > /dev/null 2>&1 &`
   (`-` fetches outputs and scores nothing locally). Outputs land in
   `runs/kaggle-fintfm-task-lite-repro-pr-bin/` and `...-mcreg/`.
6. Check each `summary.json`: every dataset `exit` is 0, `removed_fintfm_env` is `[]`, `fork`
   equals the PR head, `fintfm` is 0.11.0.
7. Untar the three `fintfm_pr_repro_*.tar.gz` into `~/proj/tabarena/experiments/`, then merge:
   `.venv/bin/python -P merge_fintfm_lite.py --name fintfm_pr_repro_all51 fintfm_pr_repro_bin fintfm_pr_repro_mc fintfm_pr_repro_reg --expect 51`
   and compare with the entry's recorded Elo using `scripts/lite_positions.py`.

Not covered: the kernels do not test that the PR branch's `FinTFM-LinProbe` defaults equal the
settings behind the recorded numbers. That is the point of the run; a difference is a finding.
