# Attention cost model for the cell-attention encoder

Openspec task 44.1 (`openspec/changes/factorized-attention`). Written 2026-11-11. Every
"measured" figure is copied from `docs/results/FINDINGS.md` (MEASURED there); every "predicted"
figure is ESTIMATED here. The tables are reproduced by `scripts/attention_cost_model.py`
(pure arithmetic, no model, no GPU).

## Result

The model retrodicts all three walls, and a fourth, within 2x. The largest miss is 1.47x
(predicted 4.4 GB against 6.5 GB measured, §81 chunked, N=2512). No discrepancy above 2x
remains, so nothing blocks 44.3.

It also says something the proposal had not yet said, and it should be read before 44.3:

* **On CUDA the quadratic term is not in memory at all.** Training memory is
  `B * N * F * d_cell * layers * ~13 * 4 bytes`, linear in every factor. The factorization
  changes the attention FLOPs (4.6x fewer per task at N=4096) but **not** this memory: a
  three-stage encoder still holds one `d_cell` token per (row, feature) in each of its cell
  layers. Measured with the existing inducing-point stage (`cell_row_attn="induced"`), the
  saved-for-backward bytes fall by 1.9% (76.46 against 77.94 units, below). At N=4096 and B=1
  the 5M model needs about 11 GB of saved activations either way; at B=8 that is 89 GB.
  Task 44.4 should expect no training-memory gain from the stage structure alone.
* **On CPU and MPS the factorization does remove the memory wall**, because attention scores
  are materialised there: 36.5 GB per layer per task at N=4096 becomes 1.1 GB.

**Status 2026-11-13.** Tasks 44.2 to 44.6 of `factorized-attention` are still open, so the
factorized encoder is not built and the section 4 projection is untested. Two later results touch
the model. DR6.2 (`cell_row_attn="induced"`, 64 points) trained 13% faster to its last
checkpoint (2,838 s against 3,267 s, §178, MEASURED, Kaggle T4x2) and scored -0.0186 mean AP
against its control, 5 of 5 folds negative, so it was not adopted. That is compatible with the
prediction above of no training-memory gain; wall time was not predicted by this model. The
2026-11-07 `cell_mixing="latent"` option (task 48.27) replaces both cell stages with latents and
is not modelled here; it has no result yet.

## 1. What the encoder does (read from `src/fintfm/modeling/model.py`)

`encode_rows` (L1399-1565 in the file as of 2026-11-13; the line numbers below were re-read on that date, and the 2026-11-11 draft's L1440-1530 had drifted). With `n_cell_blocks=K > 0` and
`cell_labels=True`, per block:

1. **Row-attention within feature**: cells reshaped to `(B*F, N, d_cell)`, attention across the
   `N` rows of one feature, `n_col_layers` pre-norm layers (`_row_within_feature`, L1145). The
   mask is a boolean `(N, N)` (`_row_mask`, L1130): every row sees the context and itself.
2. **Column attention within row**: `(B*N, F, d_cell)`, attention across features, same depth
   (`_column_attention`, L1323).
3. After the blocks the cells are pooled (mean and max) to one `d_model` vector per row, the
   label embedding is added, and `n_layers` row layers attend across the `N` rows
   (`_row_attention`).

The weights of the two cell stages are tied across blocks, so there are
`L_cell = 2 * K * n_col_layers` cell layers. Each cell layer uses `dim_feedforward = 2 * d_cell`
(L981, L999), so a cell layer is not configurable in width ratio. Dropout is 0.

| symbol | meaning | 885K shape (§79, §81, §95) | 5M shape (§94) |
| --- | --- | --- | --- |
| `F` | features | 136 | 136 |
| `d` | `d_cell` | 48 | 64 |
| `D` | `d_model` | 128 | 256 |
| `h` | `n_heads` | 4 | 4 (see below) |
| `n_col_layers` | per cell stage | 2 | 3 |
| `K` | `n_cell_blocks` | 1 | 1 |
| `n_layers` | row layers | 4 | 6 |
| `L_cell` | `2*K*n_col_layers` | 4 | 6 |

Where each value was read from:

* **885K shape**: the saved config of `runs/dl/v4-cellattn-fin10.pt`
  (`d_cell=48, d_model=128, n_heads=4, n_col_layers=2, n_layers=4, d_ff=512, n_cell_blocks=1,
  cell_labels=True, max_features=136`). §95 says "the 885K-parameter shape"; with
  `column_id_dim=16` this config has exactly 885,650 parameters (checked by building it on CPU).
* **5M shape**: §108 gives `d_cell=64, d_model=256, n_layers=6, n_col_layers=3, d_ff=1024`.
  §94 does not state `n_cell_blocks`, `n_heads` or `max_classes`. I searched
  `n_cell_blocks in {0,1}`, `n_heads in {4,8}`, `max_classes in {2,10}`, `max_features in
  {24,136}`, `column_id_dim in {None,16}`: **4,979,778 parameters (§94, §108) occurs only with
  `n_cell_blocks=1` and `max_classes=2`**. Head count does not change the parameter count and
  does not enter the CUDA model, so 4 is the default and is not load-bearing.
* **Batch and F**: §94 says batch 8; §95 says batch 4. F=136 is assumed for the §95 probe (the
  text does not state it). It is checked, not just assumed: with F=24 the prediction would be
  5.7x too small, and with F=136 it is 0.82-0.85 of the measurement at every N.
* **dtype**: float32. `TrainConfig.amp` defaults to `"none"` and no finding records another
  setting for these probes. Under bf16 every figure here halves for activations.
* **Fused or materialised**: depends on the platform, not the code. Training and inference both
  call `nn.TransformerEncoder`, whose attention goes to `scaled_dot_product_attention`.
  On CUDA that selects a flash or memory-efficient kernel (§95 reports both enabled); the score
  matrix is never stored. On CPU in no-grad eval, the stock fast path materialises
  `(B*F*h, N, N)` scores (measured below). `attn_impl` and `_prenorm_stack` (L1658) only matter for
  half precision (`_needs_aligned_heads`, L1303).

## 2. Formulas

Bytes use fp32 (4 B). `B` tasks per pass, `N = max_context + query_chunk`.

**Attention score elements, per layer**

| stage | elements | present in factorization |
| --- | --- | --- |
| row-within-feature | `B * F * h * N^2` | replaced |
| column-within-row | `B * N * h * F^2` | kept |
| row (in-context) | `B * h * N^2` | kept |
| induced column stage, `k` points | `2 * B * F * h * k * N` | new stage 1 |

**Attention FLOPs (QK^T plus AV), per layer**: `4 * F * N^2 * d` for row-within-feature,
`4 * N * F^2 * d` for column, `4 * N^2 * D` for row, `4 * F * (2kN) * d` for the induced stage.
Projection and feed-forward FLOPs are `16 d^2` per cell token per cell layer and `24 D^2` per row
token per row layer (`d_ff = 4D`), the same in both encoders.

**CPU and MPS inference (scores materialised, no grad)**

`peak ~ baseline + c * (F_chunk * h * N^2 * 4 B)`

with `F_chunk = min(feature_chunk, F)`, `baseline = 0.355 GB` and `c` score tensors alive at
once. `c` is not derived from the code. I measured it: a no-grad forward pass of the real
885K model at F=136, B=1 on this machine's CPU, one process per N, peak RSS above the
post-import baseline divided by one score tensor:

| N | peak RSS | c |
| --- | --- | --- |
| 256 | 0.76 GB | 2.84 |
| 512 | 2.11 GB | 3.08 |
| 768 | 3.57 GB | 2.50 |
| 1024 | 5.92 GB | 2.44 |

(SMOKE-TEST status: one run each, at a load average near 200, so they are a calibration of
`c`, not a result. The N=512 value, 2.11 GB, agrees with §79's 2.1 GB at N=516.) The model uses
`c` in `{2.0, 2.5, 3.0}`. This calibration at N <= 1024 means rows with N above 1024 are
out-of-sample predictions; rows at N <= 1024 are in-sample.

**CUDA training (scores fused)**

`saved = B * N * u * F * d * 4 B`, with `u` the saved-for-backward activations per cell token
in units of `d * 4 B`, summed over all layers. Peak allocated is `saved` times an overhead factor
(parameters, gradients, optimiser state, backward transients).

`u` was measured, not assumed, with `torch.autograd.graph.saved_tensors_hooks` on the real
`FinancialTFM` in train mode (CPU, B=1, N=64 and N=128, unique storages counted once, slope
between the two N so weights and other constants cancel):

| model | parameters | MB per row | `u` | `13 * L_cell` |
| --- | --- | --- | --- | --- |
| 885K | 885,650 | 1.3649 | 52.27 | 52 |
| 5M | 4,979,778 | 2.7135 | 77.94 | 78 |
| 5M, `cell_row_attn="induced"`, 64 points | 5,092,482 | 2.6621 | 76.46 | 78 |

So `u = 13 * L_cell` to within 0.5%. The 13 is `9 + 2 * (d_ff/d) ` with `d_ff/d = 2`: nine
`d`-wide tensors per layer (LayerNorm input and output, the q/k/v block, the attention output,
its projection input, the residuals) and two `2d`-wide ones. The CPU flash kernel saves the same
set CUDA's memory-efficient kernel does (q, k, v, output, log-sum-exp), and no `(N, N)` tensor
appears in the saved set on CPU either, which is why this measurement transfers. The row stage
and the embeddings are inside the 13 per layer on average; they are 3.5% of the total at the 5M
shape.

## 3. Predicted against measured

### 3.1 CPU inference, §79 and §81 (885K shape, F=136, B=1)

`c` central (2.5) and the range (2.0 to 3.0). Ratio is central predicted over measured.

| source | N | `feature_chunk` | predicted (range) GB | central GB | MEASURED GB | ratio | within 2x |
| --- | --- | --- | --- | --- | --- | --- | --- |
| §79 | 516 | none | 1.5-2.1 | 1.8 | 2.1 | 0.86 | yes |
| §79 | 1016 | none | 4.8-7.1 | 6.0 | 6.7 | 0.89 | yes |
| §79 | 2016 | none | 18.0-26.9 | 22.5 | **15.9** | 1.41 | yes |
| §81 | 2024 | none | 18.2-27.1 | 22.6 | **21.0** | 1.08 | yes |
| §81 | 2512 | none | 27.8-41.5 | 34.7 | 27.2 | 1.28 | yes |
| §81 | 2024 | 16 | 2.5-3.5 | 3.0 | 3.4 | 0.88 | yes |
| §81 | 2512 | 16 | 3.6-5.2 | 4.4 | 6.5 | 0.68 | yes |
| §81 | 3024 | 16 | 5.0-7.4 | 6.2 | 9.1 | 0.68 | yes |
| §81 | 5024 | 16 | 13.3-19.7 | 16.5 | 15.7 | 1.05 | yes |
| §81 | 6048 | 16 | 19.1-28.4 | 23.8 | 22.5 | 1.06 | yes |
| §79 | 4048 | none | 71.7-107.3 | 89.5 | 63 (ESTIMATED in §79) | 1.42 | n/a |

Notes on the rows:

* **"16 GB at N=2024" is two different measurements.** The task text, and the proposal, quote
  §79 as 16 GB at N=2024. §79's CPU table says **15.9 GB at N=2016**; N=2024 appears in §79 only
  for the MPS timing (no memory). §81, one process per configuration, measured **21.0 GB at
  N=2024** on the same checkpoint with no chunking. The two differ by 1.32x at nearly the same
  N (N differs by 0.4%, which is 0.8% in N^2). The model brackets both (18.2-27.1 GB at N=2024).
  I cannot tell from the findings why they differ; §81's own process-lifetime `ru_maxrss`
  warning is the likeliest class of cause. It is below 2x and I did not chase it further.
* **Large unchunked N sits at the low end.** Effective `c` is 1.77 at §79's N=2016 and 1.96 at
  §81's N=2512, against 2.4-3.1 at N <= 1024. A plausible reading is that at these sizes the
  Mac compresses or pages part of the score buffer, so resident size under-reports it
  (§79 itself says the activation set "stops fitting in unified memory and spills" between
  2,000 and 2,500 rows). That is a hypothesis; I did not test it.
* **§79's "~63 GB at N=4048" was a quadratic extrapolation of the 15.9 GB point**
  (15.9 x (4048/2016)^2 = 64.1 GB), so it inherits that point's low effective `c`. The model
  gives 72-107 GB. Not a measurement either way.
* **Chunked rows under-predict at N=2512 and 3024** (0.68x), over-predict at nothing. The
  floor under the score term (activations of the other stages, output arrays) grows with N and
  is not modelled; at N=6048 the score term dominates and the ratio is 1.06.

### 3.2 CUDA training, §94 and §95

Predicted is the saved-for-backward bytes; "saved/meas" is the ratio to the measured peak.

| source | model | B | N | predicted saved GB | MEASURED peak GB | saved/measured |
| --- | --- | --- | --- | --- | --- | --- |
| §95 | 885K | 4 | 128 | 0.70 | 0.85 | 0.82 |
| §95 | 885K | 4 | 256 | 1.40 | 1.67 | 0.84 |
| §95 | 885K | 4 | 384 | 2.10 | 2.49 | 0.84 |
| §95 | 885K | 4 | 512 | 2.80 | 3.31 | 0.84 |
| §95 | 885K | 4 | 768 | 4.19 | 4.96 | 0.85 |
| §94 | 5M | 8 | 512 | 11.11 | 12.28 | 0.91 |
| §94 | 5M | 8 | 1024 | 22.23 | **OOM on 23.7 GB** | see below |

* **Linearity (§95)**: the predicted doubling ratios are exactly 2.00 (the formula is linear in
  N); measured x1.97, x1.98, x1.99. The constant gap, 15-18% above `saved`, is overhead
  (parameters, gradients and optimiser state are only 0.01-0.08 GB; the rest is backward
  transients and allocator rounding), and it is stable across a 6x range of N.
* **The OOM (§94) is not predicted by first principles alone.** `saved` at N=1024 is 22.23 GB,
  94% of the card's 23.7 GB. The prediction of OOM needs the overhead: measured overhead at the
  same model's N=512 is x1.105 (12.28 / 11.11), which gives 24.6 GB, 0.9 GB over capacity. With
  §95's 15-18% instead, 25.6-26.2 GB. Both exceed 23.7 GB, but the margin is 4% on the tighter
  calibration, so the honest statement is "the model puts the wall at `n_rows` of about 990 for
  B=8 and the 5M shape, and 1024 is just past it". The proposal's "~24.5 GB by linear
  extrapolation" is the same calculation done from the measurement directly.
* **Unit check.** A 24 GB L4 cannot have 23.7 GiB available, so the 23.7 and the 12.28 are
  taken as decimal GB, matching the 1e9 used here. If they were GiB the ratios in the last
  column would be 0.84 and the OOM prediction would be unchanged in direction.
* **`feature_chunk` (§94)**: the model has no term that chunking could change on CUDA, since
  `saved` does not depend on how features are batched. §94 measured it flat to 0.2%.

### 3.3 The three walls in the proposal

| wall | platform | what the model says | verdict |
| --- | --- | --- | --- |
| §79, ~16 GB at N~2024 (15.9 at 2016; 21.0 at 2024 in §81) | CPU, materialised scores | `c * F * h * N^2 * 4` = 18-27 GB | reproduced, 1.08x to 1.41x |
| §84, `max_context=4000` unreachable (63 GB est., §79; 22.5 GB chunked, §81) | CPU, materialised scores | 72-107 GB unchunked; 19-28 GB at `feature_chunk=16`, N=6048 | reproduced, 1.06x chunked |
| §94, 5M model OOM at `n_rows=1024` | CUDA, fused | 22.2 GB saved, 24.6 GB with overhead, against 23.7 GB | reproduced, margin 4% |
| §95, linear in N on CUDA | CUDA, fused | exactly linear; overhead 15-18% | reproduced |

The task names "three measurements" without listing the third. I took them to be the proposal's
table (§79, §84, §94) and added §95, which the proposal's amendment makes mandatory. §84 is not a
memory measurement: it records that `max_context=4000` could not run unchunked; the memory
figures for that wall are §79's estimate and §81's measurements, both used above.

## 4. Projection at V4FinBench's shape (5M shape, F=136, B=1, ESTIMATED)

Three-stage factorization used for the projection: stage 1 column attention through `k = 64`
inducing points (the form already in the repository as `cell_row_attn="induced"`, DR6.2),
stage 2 attention across features within a row (the current column stage), stage 3 attention
across rows (the current row stage), with the same layer counts as today. Only stage 1's cost
differs. This is a model of the cost of the structure named in the proposal, reimplementable from
the description there; it uses no third-party code.

| N | score GB per layer, current | score GB per layer, 3-stage | GFLOP per task, current | GFLOP, 3-stage | FLOP ratio | CUDA saved GB, current | CUDA saved GB, 3-stage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 0.57 | 0.143 | 68 | 48 | 1.4 | 1.39 | 1.36 |
| 1024 | 2.28 | 0.285 | 195 | 99 | 2.0 | 2.78 | 2.73 |
| 2024 | 8.91 | 0.564 | 609 | 208 | 2.9 | 5.49 | 5.39 |
| 4096 | 36.51 | 1.141 | 2171 | 474 | 4.6 | 11.11 | 10.90 |

"Score GB per layer" is one fp32 score tensor, which is what CPU and MPS materialise (multiply by
`c` of 2-3 for the peak). The 3-stage column uses the measured 76.46/77.94 activation ratio of
the induced stage, so it is an estimate for the structure, measured only for stage 1.

What follows from the table:

1. The quadratic term in the row-within-feature stage carries a factor `F = 136`. The
   factorization removes it from the cell stage: at N=4096, 36.5 GB per layer per task becomes
   1.1 GB where scores are materialised. This is the real, platform-specific benefit.
2. Attention FLOPs fall 4.6x per task at N=4096 (2171 to 474 GFLOP). What remains is the
   stage-3 `N^2 * D` term and the linear layers, so the gain saturates near 5x for this shape.
3. CUDA training memory does not move (11.11 to 10.90 GB per task). It is `B * N * F * d` times
   the cell layers, and the factorization keeps every cell token. A change that is meant to buy
   training memory at N=4096 has to reduce one of those factors (cell layers, `d_cell`, rows
   entering the cell stages, or recompute with activation checkpointing), not just restructure
   the attention. That is a design input for 44.3's config flag and for 44.2's question of which
   stages §91 needs.
4. At B=8 and N=1024 the current encoder's 22.2 GB is the wall of §94; the same at N=4096 is
   89 GB saved activations, which no single current card holds.

## 5. What this does not establish

* The model predicts cost, not accuracy. Nothing here says the factorization works as well as
  two-way cell attention (§91 says the labelled row-within-feature mechanism is necessary;
  44.2 asks whether its attention is).
* The CUDA predictions use a CPU measurement of saved-tensor bytes. The kernel-level saved set
  is the same for flash attention on both, but I have not run the hook on a CUDA device, and
  the overhead factor (x1.105 to 1.18) is read off the measurements, not derived.
* The three-stage CUDA column is an estimate from the existing induced stage, not from a built
  three-stage encoder. 44.4 measures that.
* All CPU probes ran at a load average near 200; they calibrate `c`, they are not results.
* No platform other than CPU and CUDA was modelled. MPS (§79's 92x cliff, `CLAUDE.md`'s 78 GB
  training growth, D7) is outside this model; the materialised-score term applies there but
  the unified-memory spill does not follow from it.

## Reproduce

```bash
env -u VIRTUAL_ENV uv run --frozen python scripts/attention_cost_model.py
```

The saved-tensor measurement, for `u`:

```python
seen = {}
def pack(t):
    st = t.untyped_storage()
    seen.setdefault((st.data_ptr(), st.nbytes()), 1)
    return t
with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
    model(X, y, N // 2)          # model.train(), CPU, B=1, X of shape (1, N, 136)
saved = sum(nbytes for _, nbytes in seen)   # slope between N=64 and N=128, per row
```
