# Reproduction kernels for the TabArena-Lite entry

Five Kaggle kernels and one Mac script rerun the 51 TabArena-Lite datasets (30 binary,
8 multiclass, 13 regression; repeat 0, fold 0) from the fork's pull-request branch the way a
maintainer would: no `FINTFM_*` variable, no local checkpoint, `fintfm` from PyPI (0.12.0),
checkpoints fetched by the fork's `fintfm/checkpoints.py` from Hugging Face. Each fits the
registry entry `FinTFM-LinProbe`, default config, through `TabArenaV0pt1ExperimentBundle` and
`TabArenaContext.build_and_run_jobs(subset="lite")`, one process per dataset.

| kernel | run (result name) | machine |
| --- | --- | --- |
| `task_lite_repro_pr_bin_small_a` | `fintfm_pr_repro_bin_small_a` (binary) | Kaggle CPU |
| `task_lite_repro_pr_bin_small_b` | `fintfm_pr_repro_bin_small_b` (binary) | Kaggle CPU |
| `task_lite_repro_pr_bin_large` | `fintfm_pr_repro_bin_large` (5 binary) | Kaggle T4 |
| HF Jobs (`HF_BINARY_LARGE`) | `fintfm_pr_repro_bin_large_hf` (4 binary) | HF `t4-medium` |
| `task_lite_repro_pr_mc` | `fintfm_pr_repro_mc` (7 multiclass) | Kaggle CPU |
| `task_lite_repro_pr_reg` | `fintfm_pr_repro_reg` (6 regression) | Kaggle CPU |
| Mac (`MAC_MULTICLASS`, `MAC_REGRESSION`) | `fintfm_pr_repro_mac_mc` (SDSS17), `fintfm_pr_repro_mac_reg` (7) | Mac CPU, 4 threads |

Each table runs where its stored entry run ran. The 8 Mac tables were fitted on the Mac's CPU
(`FINTFM_DEVICE=cpu`, 4 threads); on Kaggle's 4-core CPU their first bagging fold exceeds
AutoGluon's per-fold budget and the fit stops with `TimeLimitExceeded` (2026-11-18, FINDINGS
§293). The large binary tables need a T4
for the same reason. On HF Jobs `t4-medium` fits took 1.8 to 3.3x as long as on Kaggle's T4:
4 large tables completed there and equal the stored runs within 0.01% (`HF_BINARY_LARGE`,
`scripts/hf/repro_kernel_on_hf_job.py` with `DATASETS`); bank-marketing, credit_card_clients_default
and HR_Analytics stopped with `TimeLimitExceeded` (credit_card also on `a10g-large`), and
GiveMeSomeCredit and airline were not tried, so those 5 stay in the T4 kernel.

The Kaggle scripts are generated: edit `scripts/kaggle/repro_pr_kernel_template.py` or
`make_repro_pr_kernels.py`, run `.venv/bin/python scripts/kaggle/make_repro_pr_kernels.py`
(`tests/test_repro_pr_kernels.py` fails when they are out of date, and checks that the kernels
and the Mac lists together cover the 51 datasets once).

## Preflight (the kernel exits non-zero and says why)

1. `FORK_COMMIT` is not a 7-40 digit hex SHA.
2. `fintfm==FINTFM_VERSION` cannot be downloaded from PyPI.
3. After cloning the fork at `FORK_COMMIT`: `checkpoints.py` still has an `"unpublished"`
   reference, or no reference, for a problem type the kernel runs.

## Steps

1. Update the PR branch and note its head SHA; set `FORK_COMMIT` in the template, check
   `FINTFM_VERSION` equals the release, regenerate.
2. Dry-run any kernel's preflight with internet (clones the fork, fits nothing):
   `python3 scripts/kaggle/<kernel>/<kernel>.py --preflight-only` (a Python with pip; the uv venv has none).
3. Launch one lane per kernel from the repository root, CPU kernels with
   `scripts/kaggle/launch_queue_cpu.sh`, the T4 kernel with `scripts/kaggle/launch_queue.sh`:
   `nohup scripts/kaggle/launch_queue_cpu.sh lane-mc task_lite_repro_pr_mc:- > /dev/null 2>&1 &`
   (`-` fetches outputs and scores nothing locally). Outputs land in
   `runs/kaggle-fintfm-task-lite-repro-pr-<kernel suffix>/`.
4. Run the Mac tables one at a time in a clean environment at the same fork commit, with no
   `FINTFM_*` variable. The adapter fits on CPU unless CUDA is granted (fork 65c21d5). Before
   that commit it chose Apple's MPS automatically, and every Mac table ran out of Metal memory
   within minutes under torch 2.15.0 and 2.14.0 alike (2026-11-18).
5. Check each `summary.json`: every dataset `exit` is 0, `removed_fintfm_env` is `[]`, `fork`
   equals the PR head, `fintfm` is the release.
6. Untar the `fintfm_pr_repro_*.tar.gz` into `~/proj/tabarena/experiments/`, merge the seven
   runs with `merge_fintfm_lite.py --name fintfm_pr_repro_all51 ... --expect 51`, and score
   against the stored entry with `scripts/lite_same_fit.py` and `scripts/lite_positions.py`.

Not covered: the kernels do not test that the PR branch's `FinTFM-LinProbe` defaults equal the
settings behind the recorded numbers. That is the point of the run; a difference is a finding.
