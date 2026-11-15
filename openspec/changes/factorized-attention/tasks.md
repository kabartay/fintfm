# Tasks

- [x] 44.1 **Done (2026-11-11, docs/design/attention_cost_model.md).** **Write the cost model down and check it against the three measurements.** Derive
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
- [x] 44.3 **Done (2026-11-15): `cell_mixing="factorized"` (existing `--cell-mixing` flag and `model.cell_mixing` key, default stays `attention`, so no new config key and no arm-file change). It replaces only each cell block's row-within-feature stage with `FactorizedColumnBlock`: a learned query pools each column's context cells into a column latent (one query against `n_ctx` keys per column, linear in rows), one pre-norm layer mixes latents across columns, and an MLP adds the latent back to the column's cells; the existing column-within-row attention, row pooling and in-context row encoder then give the three stages. Requires `n_cell_blocks >= 1`, `cell_row_attn="full"`, `readout="pool"`. Tests in `tests/test_model.py` (`test_factorized_*`) pin column equivariance of the block and invariance of the model, invariance to context-row order, no query-to-row leakage and exact chunked scoring, no (N, N) cell attention, a strict default state_dict load, a factorized checkpoint round trip, and a 3-step training on CPU (plus MPS/CUDA when present); golden outputs and the pre-lever fingerprints pass unchanged. Parameter count at the tiny test shape (`_P5_BASE`, 2 blocks, labels): 23,331 attention, 26,115 factorized. SMOKE-TEST only, not 44.4's measurement and not a claim about accuracy or cost at scale: forward pass on CPU (2 threads, batch 2, 32 features, `d_cell=32`, 1 block, N=512, 384 context rows, median of 3 after one warm-up, machine under load average ~190), attention 0.107 s against factorized 0.041 s; peak RSS was 356 MB against 362 MB and is dominated by the torch import, so it says nothing about activation memory. 44.2 is still open: whether this column latent carries what §91 needs is unmeasured.** **Implement the factorized encoder** behind a config flag, defaulting off so every
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
