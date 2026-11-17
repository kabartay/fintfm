# Tasks

- [x] 32.1 Replicate §29 across three seeds on the survival path before anything else. Verify:
      `uv run fintfm-ctxsweep --seeds 0,1,2`; if the ordering does not hold, §29 is; if the ordering does not hold, §29 is
      downgraded and this proposal closes.
- [x] 32.2 Add the strategy sweep to the binary benchmark across all four panels, correction on.
      Verify: `uv run fintfm-bench` reporting a strategy column per dataset.
- [ ] 32.3 **Progress (2026-11-15, binary path): `fintfm-binctx` (`src/fintfm/experiments/binary_context_sweep.py`) is the binary-path counterpart: strategies uniform, balanced, hybrid, prototype, retrieval at context sizes 500, 1000, 2000, seeds 0, 1, 2, rank transform on, Polish 1y/3y/5y and Taiwan, per-cell JSON with AUC, AP, ECE and the slice default rate, and a paired bootstrap for every strategy pair on the shared held-out rows, Holm-adjusted across the whole record separately for AUC and AP. Tested on synthetic panels only (`tests/test_binary_context_sweep.py`); a SMOKE-TEST of 8 cells at 600 rows took about 17 s wall, which says nothing about the strategies. The measurement is still owed (queue it): `env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 15 env -u VIRTUAL_ENV uv run --frozen fintfm-binctx --model <checkpoint with at least 95 features> --out runs/binary-context-sweep > run.log 2>&1`; defaults give 180 cells (`--n-ensemble 8 --n-boot 2000`), and `binary_context_sweep.json` lands in `--out`. The survival-path note follows.** **Progress (2026-11-15): the harness is ready, the record is not. `fintfm-ctxsweep` now keeps each cell's curve and records, for every strategy pair at every context size, seed and scored horizon, the company-clustered paired bootstrap difference, interval, raw p and Holm-adjusted p (one family across the record), via the shared `paired_auc_difference` and `holm_adjusted_p`; the summary prints them and `--n-boot` sets the resamples. Tested on synthetic arrays only (`tests/test_holm_harnesses.py`). Still owed: a sweep run with this code and the adjusted p-values written beside every AUC difference quoted in `FINDINGS`.** Paired bootstrap with Holm-Bonferroni over the strategy comparisons. Verify: adjusted
      p-values recorded beside every AUC difference quoted.
- [ ] 32.4 **Progress (2026-11-15, binary path): the sweep this needs now has a command. `fintfm-binctx` stores per-cell AUC, AP and the slice default rate, with uniform and balanced at the same context size, over four panels whose default rates differ (Polish 1y/3y/5y and Taiwan). After the run: `env -u VIRTUAL_ENV uv run --frozen python scripts/gap_vs_default_rate.py --binary-sweep runs/binary-context-sweep/binary_context_sweep.json` prints, per (panel, size), the seed-mean uniform minus balanced gap in AP and AUC against the slice rate, and the Spearman correlation with the number of rows behind it (`--gap-a`/`--gap-b` pick other strategies). Four panels at three sizes give at most 12 rows, so the correlation is stated with that n; it does not fit a relationship across portfolios beyond it. Not yet run; the task stays open.** **Progress (2026-11-15): not answerable from stored outputs. Only single-portfolio slices exist (V4FinBench folds span 0.3476% to 0.3698%; horizons 0.078% to 0.475%), and the binary sweeps of §35 left no per-cell outputs. On the horizon slices uniform minus balanced is negative and grows with rate (Spearman -0.84, n = 12), confounded with horizon length, so it is neither a hit nor a valid miss. Numbers and command are in `docs/results/drafts/32.4_2026-11-15.md`; the run needed is the multi-panel binary strategy sweep 32.3 also owes.** Test the majority-class mechanism: gap against portfolio default rate. Verify: the
      relationship plotted and stated numerically, with the prediction marked hit or miss.
- [x] 32.5 Resolve D9 either way, in `docs/design/DECISIONS.md`, and change the class default if the
      evidence supports it. Verify: `uv run pytest -q` green, and the default's justification in
      `classifier.py`'s docstring updated to cite the measurement rather than the paper.

**Resolved 2026-09-09.** 32.1 replicated §29 on three seeds (§33): retrieval's worst seed
beats uniform's best at every context size, and the blind ordering holds. 32.2 measured the
binary path on Polish and Taiwan (§35) and found the strategies **nearly tied** there —
uniform − balanced is −0.0004 (not significant) on Polish and +0.0077 on Taiwan — so §29's
10-12 point margin does **not** generalise. 32.5 changed the class default to `uniform` and
recorded D10.

**The mechanism was narrower than §29 proposed:** "balanced" is not one operation. The smaller
panels lack the positives for it to reach 50/50, so it lands near 25% and behaves like hybrid.
The effect scales with **how extreme the rebalancing actually is**, not with the strategy's
name.

Task 32.3's Holm-corrected bootstrap was run for the retrieval-versus-balanced comparison
(significant on both panels) but **not** for every pairwise strategy comparison, so it stays
open. 32.4's explicit gap-against-base-rate regression also stays open — §35 establishes the
direction from two panels rather than fitting the relationship.

- [x] 32.6 **Done (2026-11-17, FINDINGS §288): all 180 cells scored on Kaggle CPU in 12 kernels and merged; `retrieval` has the highest mean AP in 12 of 12 panel-size cells and `uniform` the best ECE in 11; the Holm-corrected paired bootstrap ran but cannot reject (2,000 resamples, 360 comparisons, floor 0.18); the class default is explicitly left `uniform` in DECISIONS D10. A rerun able to reject is 32.7.** **Progress (2026-11-17, FINDINGS §273): the Kaggle sweep stopped at its 11 h limit after 69 of 180 cells with no JSON, so no bootstrap or Holm result; on Polish 1y `prototype` beats `uniform` on AP in 3 of 3 seeds at every size (point values). Rerun as one kernel per panel.** **Progress (2026-11-15): command ready, measurement not run.** `env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 15 env -u VIRTUAL_ENV uv run --frozen fintfm-binctx --model <checkpoint with at least 95 features> --out runs/binary-context-sweep > run.log 2>&1` runs `prototype` beside uniform, balanced, hybrid and retrieval on Polish 1y/3y/5y and Taiwan, three seeds, and prints mean AP, AUC and ECE per strategy plus Holm-adjusted paired bootstrap tables for AP and AUC; ECE is a per-cell mean, not bootstrapped. The `docs/design/DECISIONS.md` entry on the class default is still owed after the run. Measure `prototype` on the **binary** path across Polish and Taiwan, three seeds,
      before it becomes the class default. §38 establishes it as best on the V4FinBench
      survival path only, and D9's lesson was that a context result on one path does not
      transfer to the other — §35 found the blind strategies nearly tied on the binary panels
      because "balanced" there cannot reach 50/50. Verify: AUC and ECE against `uniform` and
      `retrieval` with a Holm-corrected paired bootstrap, and the class default changed or
      explicitly left alone in `docs/design/DECISIONS.md`.

- [ ] 32.7 **Progress (2026-11-17): code ready, measurement not run.** The sweep stores per-cell probabilities and per-split labels, restricts the family with `--compare-against uniform`, recomputes from a record with `--recompare`, and refuses a bootstrap whose Holm floor is at or above 0.005 (default `n_boot` 20,000; `--family-size 72` for per-panel kernels). The Kaggle rerun and its pre-registration are next. Rerun the binary context comparison so its test can reject (FINDINGS §288). §288's
      merged Holm family of 360 comparisons per metric at 2,000 resamples has an adjusted-p floor
      of 0.18. Pre-register a smaller family (`uniform` against `prototype` and `retrieval` only,
      AP and AUC) and size `n_boot` so the floor is below 0.005; keep each cell's held-out
      labels and predictions in the record so tests can be redone without scoring again.
      Verify: a merged record whose summary prints no Holm-floor warning, and D10 updated with
      the corrected result.
