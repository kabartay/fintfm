# Tasks

- [x] 43.1 **DONE (§88). The cheap gate: measure marginal invariance on existing checkpoints.** Warp the
      Bayes-ceiling task's features with strictly-monotone maps (power p=3 and p=1/3,
      exponential, logistic) and score `v4-cellattn-fin10` and `v4-colid-fin07` both raw and
      rank-transformed. The Bayes-optimal AUC is identical under every warp, so achieved-AUC
      differences are pure non-invariance. Verify: the run asserts Spearman correlation > 0.9999
      between warped and unwarped values of the informative column, so a "warp" that silently
      destroyed rank information cannot be mistaken for a model failure. Done — §88's table.
- [x] 43.2 **DONE (§88): outcome (c). Route on 43.1's outcome, written down before the numbers arrive.** Three cases:
      (a) raw collapses and rank-transform does not rescue it → the model is not
      marginal-invariant, 43.3 is justified; (b) raw holds → the model is already invariant,
      §87 is a measured curiosity, **close this proposal without spending GPU**; (c) raw
      collapses but rank-transform rescues it → the transform is doing the invariance work, and
      the cheaper fix (match transforms in training) is preferred over augmentation. Verify: the
      finding that reports 43.1 names which of (a), (b), (c) occurred, and the proposal's status
      is set accordingly rather than defaulting to "proceed". Done — §88 names outcome (c):
      the rank transform pins performance flat across every warp and is already the inference
      default, so the augmentation's only remaining value is the gap between rank-clamped and
      best-raw performance, which is ~0.001 on the cell-attention checkpoint. Tasks 43.3-43.5
      and 43.8 are CLOSED UNMEASURED on that basis.
- [x] 43.3 **CLOSED UNMEASURED (§88, outcome (c)) — Implement random monotonic marginal augmentation** — per column, per task: rank
      the values, then map through the inverse CDF of a randomly drawn target marginal (normal,
      uniform, lognormal, Student-t, and a random monotone spline), so arbitrary marginal shape
      is spanned. Off by default, enabled by a training flag, so every existing checkpoint
      reproduces unchanged. Verify: a test asserts the augmentation preserves each column's
      rank ordering exactly (Spearman 1.0 up to ties), so it provably cannot change a task's
      label-relevant information.
- [x] 43.4 **CLOSED UNMEASURED (§88, outcome (c)) — Score the augmented checkpoint on the Bayes-ceiling probe FIRST.** A prior change
      that degrades basic signal extraction must be caught by the instrument built for exactly
      that (§74), before any real-data claim is attempted. Verify: regret at targets
      0.900/0.990/0.999 is reported against the unaugmented checkpoint at matched
      `column_id_dim`, per §86's flagged confound.
- [x] 43.5 **CLOSED UNMEASURED (§88, outcome (c)) — Only then V4FinBench**, five folds, paired bootstrap, Holm-corrected, full
      1.0M-row panel, against the current best checkpoint at matched context. Verify: per-fold
      sign counts are reported beside every difference, since §84 established that at ~200k
      rows per fold every comparison clears significance including ones too small to act on.
- [x] 43.6 **Moot while 43.3 is closed, kept as standing discipline.** The `cell_labels` ablation (runs A/B, 2026-09-17)
      is the control for 43.5. Whatever it establishes about `cell_labels` must be held FIXED in
      the augmentation run. Verify: the augmentation run's config differs from its control in
      the augmentation flag alone — this is the specific error that made §78/§80's +0.049
      unattributable and cost two GPU runs to repair.
- [x] 43.7 **DONE (§89) — the clip costs nothing.** `normalize_features` clips z-scores at ±10, and
      §87 measured 0.029% of training values hitting it against 0.000% of rank-transformed
      ones. Almost certainly minor, and free to check on existing checkpoints. Verify: achieved
      AUC on the ceiling probe is reported at the current clip and at a wider one, on a
      checkpoint that already exists, with no retraining. Done — widening 10 to 100 never
      helps and is mildly worse on the heaviest tails (-0.0015), consistent across all six
      cells. No change made.
- [x] 43.8 **CLOSED UNMEASURED (§88, outcome (c)) — Decide the inference default deliberately.** If the model becomes marginal-
      invariant, `feature_transform="rank"` may stop earning its +0.086 (§35) and the default
      should be re-measured rather than inherited. Verify: §35's comparison is re-run on the
      augmented checkpoint and the packaged default in `src/fintfm/configs/default.yaml` is
      either changed or explicitly reaffirmed with the new number.
