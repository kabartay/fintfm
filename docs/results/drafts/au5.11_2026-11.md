# AU5.11: prior statistics, before and after the existing flags (2026-11-04)

Draft; not a FINDINGS entry. No prior code or default changed.

## Result

All numbers MEASURED with `scripts/prior_stats.py` logic (a4806e2), 300 tasks per mixture,
seed 0, row counts 256/512/1024, `max_features` 136, `max_classes` 10, CPU, 2 threads.
Raw output: `runs/agents1104/AU5.11/prior_stats.txt`; driver `au511_driver.py` beside it
(adds one mixture, `broader_flags`, without editing the script).

Same-session reproduction check: the `shipped` mixture (p_financial 1.0) gives label-informative
missingness in 11.7% of binary tasks and median excess kurtosis 4.06, identical to the figures
in AU5.11, so the "before" is reproducible.

Mixtures: `default` = PriorConfig defaults (financial 0.7, SCM 0.3); `shipped` = p_financial
1.0; `broader` = p_financial 0.3, p_tree 0.3, p_task_family 0.3 (the arm, no new flags);
`broader_flags` = `broader` plus `categorical_mode="unordered"`, `missing_mode="mixed"`,
`feature_ratio_bands` (0.5,0,0.05),(0.35,0.05,0.2),(0.15,0.2,0.5) (the V2_KNOBS values).
Realised sources, broader_flags: scm 105, tree 89, family 66, financial 40.

| statistic (300 tasks) | default | shipped | broader | broader_flags |
| --- | --- | --- | --- | --- |
| binary share of tasks | 0.703 | 1.000 | 0.427 | 0.410 |
| minority share p5/p50/p95 | 0.00195/0.0391/0.25 | 0.00391/0.0327/0.223 | 0.00195/0.125/0.495 | 0.00195/0.141/0.496 |
| tasks with a class of <= 2 rows | 0.133 | 0.080 | 0.170 | 0.163 |
| features p5/p50/p95 | 7.95/70/130 | 11/74/132 | 14/77.5/136 | 2.95/31.5/136 |
| categorical column share, mean | 0.0685 | 0.0122 | 0.1242 | 0.1298 |
| tasks with any categorical | 0.847 | 0.820 | 0.730 | 0.497 |
| categorical levels p5/p50/p95 (max) | 2/5/9 (12) | 2/7.5/12 (12) | 2/5/7 (12) | 2/10/54 (64) |
| missing cells, mean share | 0.0676 | 0.0780 | 0.0492 | 0.0283 |
| tasks with any missing | 0.840 | 1.000 | 0.497 | 0.527 |
| binary tasks, label-informative missingness | 0.085 | 0.117 | 0.047 | 0.057 |
| excess kurtosis p5/p50/p95 | -1.21/0.46/229 | -1.21/4.06/333 | -1.18/-0.032/15.5 | -1.19/-0.026/34.8 |
| kurtosis share > 3 / > 10 | 0.403/0.305 | 0.520/0.405 | 0.082/0.057 | 0.097/0.070 |

Reading: moving to the arm mixture cuts the label-informative share from 11.7% to 4.7% (the
financial prior is the only source that produces it; its share falls from 1.0 to 0.147) and
the median kurtosis from 4.06 to -0.03. The flags leave the kurtosis median unchanged (share > 3 rises from 0.082 to 0.097; cause not investigated) and do not make missingness label-informative: 5.7% against
the 1% expected under MCAR with Bonferroni, so MAR/MNAR via feature correlation does leak into
the label somewhat, but never by design. The 4.7% and 5.7% are not distinguishable from each
other at n = 300 (about 120 binary tasks each); not tested.

## Gap status (current code, main 115535f)

| gap | status | evidence |
| --- | --- | --- |
| categoricals always ordered 2-7 level codes | closed by flag, default still ordered | `prior/scm.py:263-269` and `prior/tree.py:158-163` still draw `integers(2, 8)` unless `knobs.unordered`; `categorical_mode="unordered"` (`mixture.py` field, `knobs.py:432` `_unordered_code`) gives p95 54 levels, max 64 |
| missingness only MCAR, never label-informative | MCAR half closed by flag (`missing_mode="mixed"`, `knobs.py:494 _missing_mask`: MCAR/MAR/MNAR); label-informative is not a design choice in any general source | MAR/MNAR drive from features only; measured 5.7% vs 4.7% |
| SCM/tree marginals near-Gaussian | still present | no flag draws heavy-tailed marginals; `tie_column_prob` (`knobs.py:460`) adds ties, point masses, zero inflation only. Medians -0.03 (broader), -0.03 (broader_flags) |
| task-family generator always 136 Gaussian columns | width closed by flag; Gaussian still present | `mixture.py:850-852` passes `width` from `feature_ratio_bands` when set, else `max_features`; families still draw `rng.normal` (`task_families.py:105,148,188,275`). Not separately measured per source for family width |
| tree splits pick columns by raw std | still present | `prior/tree.py:82-84` (`spread = X.std(axis=0)`, `p = spread/total`) |

Original line references have moved: SCM categoricals were `scm.py:261-264` (now 263-269), family
width `mixture.py:229-236` (now `_sample_task_family_mixture` at 69-86 and 850-852), tree
std `tree.py:79-82,136` (now 82-84).

AU1.13 is done (tasks.md:126), so the arm's stated ordering prerequisite is met. The `Verify`
clause (statistics before and after) is met for the flag-only arm; open for any code change below.

## Proposed smallest code changes for the open gaps (not implemented)

1. Heavy-tailed marginals. Add `PriorConfig.heavy_tail_column_prob: float = 0.0` and, in
   `apply_task_knobs` (`prior/knobs.py`, next to the tie block at ~line 575), for each numeric
   column with that probability apply a monotone signed transform of the standardised column,
   `sinh(a*z)/a` with `a` drawn LogU(0.3, 2) or exp of z with a drawn sigma; monotone so the
   label relation is kept in rank. Default 0.0 consumes no draw. Wire as a `TaskKnobs` field and
   `check_knobs` entry, plus the train.py flag list at `modeling/train.py:223`.
2. Tree splits by raw std. In `prior/tree.py:82-84` add a `split_by: str = "std"` argument,
   `"uniform"` giving `p = ones/n_features`, defaulting to current behaviour; expose through
   `TaskKnobs` or `PriorConfig.tree_split_uniform_prob`. Rationale: raw std picks scaled columns
   preferentially, so unstandardised wide columns dominate; uniform draw needs no extra RNG draw
   beyond the existing `rng.choice`.
3. Label-informative missingness in general priors. Add a `MISSING_MECHANISMS` entry `"mnar_label"`
   in `prior/knobs.py` (`_missing_mask` takes `y`; the call at the `mixed_missing` block passes
   `task.y`), using the label (or a feature-score proxy) as the driver `z` in the existing
   logistic mask. Only the 4.7% to 11.7% gap in the arm measures this, and the financial prior
   still supplies it at 0.147 share; do this only if the arm result motivates it.
4. Gaussian-only families: depends on 1; applying heavy-tail transform in `apply_task_knobs`
   already covers `task_family` (in `_KNOB_SOURCES`, `mixture.py:416`).

## Open

- The arm checkpoint itself (training) is not run here.
- Per-source kurtosis and family width distribution under bands not measured separately.
- Whether 4.7% vs 5.7% differ is untested.
