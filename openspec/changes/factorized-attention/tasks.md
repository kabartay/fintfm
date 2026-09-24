# Tasks

- [ ] 44.1 **Write the cost model down and check it against the three measurements.** Derive
      attention cost for the current encoder and for a three-stage factorization at
      V4FinBench's shape (136 features, N up to 4096), and confirm the derivation reproduces
      §79's measured 16 GB at N=2024 and §94's OOM at `n_rows=1024` for 5M parameters. Verify:
      the predicted and measured figures are reported side by side, and a discrepancy larger
      than 2x is resolved before any code is written — a cost model that cannot retrodict
      three known walls cannot be trusted to predict a fourth.
- [ ] 44.2 **Establish what §91's mechanism actually requires.** The unlabelled variant scores
      below chance, so per-cell label injection is necessary; what is unknown is whether the
      *row-attention-within-feature* stage is necessary or whether a cheaper column summary
      (per-column moments, a learned pooled statistic) carries the same information. Verify:
      an ablation replacing row-within-feature attention with a non-attentional per-column
      summary is scored on the Bayes-ceiling probe and the antisymmetric probe, against both
      the current architecture and the `n_cell_blocks=0` baseline.
- [ ] 44.3 **Implement the factorized encoder** behind a config flag, defaulting off so every
      existing checkpoint reproduces unchanged. Verify: a test asserts the new path is
      permutation-equivariant in columns and permutation-invariant in context rows, the two
      invariances `docs/design/DECISIONS.md` D4 requires and which the current encoder's tests
      already pin.
- [ ] 44.4 **Measure cost before measuring accuracy.** Verify: peak training memory and
      seconds per step are reported for both encoders at matched shape, and the factorized
      path is shown to train 5M parameters at `n_rows=1024` on a 24 GB card — the specific
      thing §94 measured as impossible — before any accuracy comparison is attempted.
- [ ] 44.5 **Then accuracy, at the project's standard.** Verify: five-fold V4FinBench with
      paired bootstrap and per-fold sign counts, plus the Bayes-ceiling probe, against the
      current architecture at matched `column_id_dim` and matched task count.
- [ ] 44.6 **Re-test the levers this unblocks.** `max_context` past 2,000 and model size past
      5M parameters were both ruled unreachable rather than unhelpful (§84, §94). Verify: if
      44.4 succeeds, the context sweep and the size ladder are re-run at the newly reachable
      settings, and §83/§93's nulls are explicitly re-checked rather than assumed to carry
      over.
