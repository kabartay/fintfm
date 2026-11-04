# AU5.11 part 2: the three prior-diversity flags, implemented (2026-11-04)

Draft; not a FINDINGS entry. Code on branch `worktree-agent-ab911f54f4979e989`. Every flag
defaults off and leaves generated tasks byte-identical to `63500ad`.

## What was added

| gap (draft au5.11_2026-11) | flag | where | on |
| --- | --- | --- | --- |
| near-Gaussian marginals | `heavy_tail_column_prob` (float, 0.0) | `prior/knobs.py` `_heavy_tail_column`, called in `apply_task_knobs` before the tie block | each numeric column, with this probability, goes through `sinh(a z) / a`, `z` standardised on finite values, `a` LogU(0.3, 2) |
| tree splits by raw std | `tree_split_by` (`"std"` / `"uniform"`, `"std"`) | `prior/tree.py` `_oblivious_forest(split_by=...)` | `"uniform"`: `p = 1/n_features` |
| missingness never label-driven | `missing_label` (bool, false) | `prior/knobs.py` `_missing_mask(rng, X, y=None)` | adds `"mnar_label"` to the mechanism table when `y` is given |

Why `sinh(a z)/a`: odd, strictly increasing (ranks, ties, point masses and NaN survive, so the
label's relation to the column keeps its order); linear near 0, exponential in the tails;
one parameter. A Student-t quantile map was the alternative; it needs a distribution table
and gives no extra property the tests need.

`mnar_label`: each class gets a N(0,1) score, the row's score is standardised, and the
existing logistic mask (slope U(1,3) with random sign, intercept bisected to the drawn rate)
is driven by it. It applies only where a classification `y` exists (`label_knobs` true and no
`y_continuous`). `MISSING_MECHANISMS` itself is unchanged, so the default draw
`rng.integers(3)` is too. `missing_label` without `missing_mode="mixed"` is refused by
`check_knobs`.

Plumbing: `TaskKnobs`, `knobs_from`, `check_knobs`; `PriorConfig` and
`config.PriorConfigDefaults` fields; `configs/default.yaml` keys; `modeling/train.py`
`_PRIOR_CONFIG_KNOBS` and flags `--heavy-tail-column-prob`, `--tree-split-by`,
`--missing-label`. The existing drift guard
(`tests/test_config.py::test_inference...` / prior-section test iterating
`_PRIOR_CONFIG_KNOBS`) covers the new keys without edits; `load_config` validates
`tree_split_by` and `heavy_tail_column_prob`.

## Tests (`tests/test_prior_diversity.py`, 13 tests, passing)

- Off is byte-identical: a SHA-256 over 40 tasks from each of two mixtures (plain; and with
  the DR2.3 knobs on) at seed 2026 is `45068bcb...1caa` on a checkout of `63500ad` and on this
  branch (MEASURED, both on darwin arm64); pinned on that platform, and flags-stated-off equals
  flags-omitted on every platform.
- Heavy tails: Spearman rank correlation with the input is 1.0; excess kurtosis of a 4000-point
  Gaussian rises in 20 of 20 draws, median gain above 1; NaN, ties, constants untouched; and
  through `sample_task` the median column kurtosis at prob 1.0 exceeds prob 0.0 by more than 1.
- Tree splits: with one column scaled by 1000, `"std"` gives it probability above 0.99 and
  `"uniform"` gives every column 0.2; unknown values refuse.
- Missingness: with features independent of `y`, `mnar_label` gives a median class gap in
  blanked share above 0.15, MCAR and MNAR below 0.1; without `y` the table and stream are
  unchanged.
- Config and CLI reach `PriorConfig`; bad values refuse.
- `tests/test_prior_diversity.py tests/test_config.py tests/test_prior_v2.py
  tests/test_tree_prior.py`: 114 passed (first run, before the last test-file lint
  refactor; the new file alone re-ran at 13 passed). `ruff check src/ tests/ scripts/` clean.

## prior_stats (MEASURED)

300 tasks per mixture, seed 0, row counts 256/512/1024, `max_features` 136, `max_classes` 10,
CPU, `OMP_NUM_THREADS=2`, nice 19. Driver `au511b_driver.py` (copied beside the raw output),
raw output `runs/agents1104/AU5.11b/prior_stats.txt`.
`broader_flags` = the earlier draft's mixture (`broader` plus `categorical_mode="unordered"`,
`missing_mode="mixed"`, V2 bands). `broader_flags_new` = the same plus
`heavy_tail_column_prob` 0.15, `tree_split_by` `"uniform"`, `missing_label` true.
The off column reproduces the earlier draft's `broader_flags` column exactly.

| statistic (300 tasks) | broader_flags (off) | broader_flags_new (on) |
| --- | --- | --- |
| realised sources | scm 105, tree 89, family 66, financial 40 | scm 107, tree 86, family 68, financial 39 |
| binary share of tasks | 0.410 | 0.443 |
| minority share p5/p50/p95 | 0.00195/0.141/0.496 | 0.00195/0.11/0.496 |
| tasks with a class of <= 2 rows | 0.163 | 0.153 |
| features p5/p50/p95 | 2.95/31.5/136 | 2/30/136 |
| categorical column share, mean | 0.1298 | 0.1349 |
| tasks with any categorical | 0.497 | 0.510 |
| categorical levels p5/p50/p95 (max) | 2/10/54 (64) | 2/10/52 (64) |
| missing cells, mean share | 0.0283 | 0.0287 |
| tasks with any missing | 0.527 | 0.530 |
| binary tasks, label-informative missingness (Fisher, Bonferroni 0.01) | 0.057 | 0.090 |
| excess kurtosis p5/p50/p95 | -1.19/-0.0257/34.8 | -1.18/0.00638/64.8 |
| kurtosis share > 3 / > 10 | 0.097/0.070 | 0.151/0.107 |
| numeric columns counted | 13675 | 13128 |

Reading:

- Heavy tails: kurtosis share above 3 rises from 0.097 to 0.151 and the p95 from 34.8 to 64.8.
  The median barely moves (-0.026 to 0.006) because only 15% of numeric columns are mapped.
  The kurtosis statistic is pooled over columns, so this does not say how many tasks gain a
  heavy-tailed column.
- Label-informative missingness: 0.057 to 0.090 of binary tasks. At about 120 binary tasks per
  arm this is roughly 7 versus 11 tasks; not tested for a difference. It is below the 0.117 of
  the shipped financial-only mixture. The flag only acts in the share of tasks with missingness
  (about half) and one mechanism in four, and a task's label effect must survive the
  Bonferroni test.
- Uniform splits are not visible in this table: no statistic here measures which columns a
  tree used. The test above is the only evidence for the split change.
- Mixture rows differ slightly between arms (binary share, source counts) because
  `tree_split_by="uniform"` and the extra heavy-tail draws change the random stream after the
  first task; the arms are not paired draws.

## Open

- No training run; whether any of the three flags helps a checkpoint is untested.
- Per-task heavy-tail incidence, and the effect of uniform splits on tree-task accuracy of a
  fixed learner, not measured.
- The 0.057 versus 0.090 difference is untested at n = 300.
- `heavy_tail_column_prob` 0.15 is the tie flag's value, not tuned.
- No kaggle `ARM_FLAGS` or `scripts/kaggle` file was changed; no arm uses the new flags yet.
