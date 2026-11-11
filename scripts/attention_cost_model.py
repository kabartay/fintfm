"""Attention cost model for the cell-attention encoder (openspec task 44.1).

Pure arithmetic: no model is built, no tensor is allocated, no GPU is touched. It prints three
tables (all derivations and the provenance of every constant are in
``docs/design/attention_cost_model.md``):

1. **CPU inference retrodiction**: peak resident memory of a no-grad forward pass, where
   attention scores are materialised, against the figures measured in ``docs/results/FINDINGS.md``
   §79 and §81.
2. **CUDA training retrodiction**: peak allocated memory of a forward and backward pass, where
   fused attention keeps the score matrix out of memory, against §94 and §95.
3. **Projection** at V4FinBench's shape (136 features, N up to 4096) for the current encoder
   and a three-stage factorization.

Every "measured" column is copied from FINDINGS.md (MEASURED there); every "predicted" column is
ESTIMATED here.

Run with ``uv run --frozen python scripts/attention_cost_model.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Bytes per float32 element. Both measured regimes ran in float32 (``TrainConfig.amp="none"``).
FP32 = 4

#: Score tensors live at the same time in the CPU no-grad path, as a (low, central, high) range.
#: Read from a probe of the real ``FinancialTFM`` at F=136, N=256..1024: 2.84, 3.08, 2.50, 2.44
#: (peak RSS above the post-import baseline, divided by one fp32 score tensor).
CPU_SCORE_COPIES = (2.0, 2.5, 3.0)

#: Resident memory of the interpreter, torch and the 885K-parameter model before the forward
#: pass, in GB. Probe value: 0.353-0.357 GB.
CPU_BASELINE_GB = 0.355

#: Saved-for-backward activations per cell token and cell layer, in units of
#: ``d_cell * 4`` bytes. ``13 = 9 + 2 * (d_ff / d_cell)`` with ``d_ff = 2 * d_cell`` in every
#: cell layer. Measured as 77.94 / 6 layers (5M model) and 52.27 / 4 layers (885K model) with
#: ``torch.autograd.graph.saved_tensors_hooks``, so the row stage and embeddings are folded in.
CELL_UNITS_PER_LAYER = 13.0

#: Measured total units for the two model shapes (all stages included).
UNITS_5M = 77.94
UNITS_885K = 52.27

#: The same measurement with ``cell_row_attn="induced"`` (64 inducing points).
UNITS_5M_INDUCED = 76.46


@dataclass(frozen=True)
class Shape:
    """Architecture hyper-parameters the cost depends on.

    Attributes:
        name: Label for the tables.
        features: ``F``, the feature width (V4FinBench: 136).
        d_cell: Width of a cell token.
        d_model: Width of a row token.
        heads: Heads in the cell stages (``n_heads``).
        n_col_layers: Layers per cell stage.
        n_cell_blocks: ``K``, the number of two-way blocks.
        n_layers: Row (in-context) layers.
    """

    name: str
    features: int
    d_cell: int
    d_model: int
    heads: int
    n_col_layers: int
    n_cell_blocks: int
    n_layers: int

    @property
    def stage_layers(self) -> int:
        """Layers in each of the two cell stages (``K * n_col_layers``)."""
        return self.n_cell_blocks * self.n_col_layers


#: §79, §81 and §95's 885K-parameter checkpoint (``v4-cellattn-fin10`` config).
SHAPE_885K = Shape("885K", 136, 48, 128, 4, 2, 1, 4)
#: §94's medium model (4,979,778 parameters; ``max_classes=2`` reproduces the count).
SHAPE_5M = Shape("5M", 136, 64, 256, 4, 3, 1, 6)


def score_bytes_row_within_feature(s: Shape, n: int, batch: int = 1, chunk: int | None = None) -> float:
    """Bytes of one fp32 score tensor of the row-within-feature stage.

    Args:
        s: Architecture.
        n: Rows per task, ``N = max_context + query_chunk``.
        batch: Tasks per forward pass.
        chunk: Features processed together (``feature_chunk``); ``None`` means all of them.

    Returns:
        ``batch * min(chunk, F) * heads * N^2 * 4`` bytes.
    """
    f = s.features if chunk is None else min(chunk, s.features)
    return float(batch * f * s.heads * n * n * FP32)


def cpu_peak_gb(s: Shape, n: int, chunk: int | None, copies: float) -> float:
    """Predicted peak resident memory of a no-grad CPU forward pass, in GB.

    Args:
        s: Architecture.
        n: Rows per task.
        chunk: ``feature_chunk`` or ``None``.
        copies: Score tensors alive together (see :data:`CPU_SCORE_COPIES`).

    Returns:
        Baseline plus ``copies`` score tensors, in decimal GB.
    """
    return CPU_BASELINE_GB + copies * score_bytes_row_within_feature(s, n, 1, chunk) / 1e9


def cuda_saved_gb(s: Shape, batch: int, n: int, units: float) -> float:
    """Saved-for-backward activation bytes of one training step, in decimal GB.

    Args:
        s: Architecture.
        batch: Tasks per step.
        n: Rows per task.
        units: Total units per cell token (:data:`UNITS_5M`, :data:`UNITS_885K`).

    Returns:
        ``batch * N * units * F * d_cell * 4`` bytes: linear in every factor, no ``N^2``.
    """
    return batch * n * units * s.features * s.d_cell * FP32 / 1e9


def attention_gflop_current(s: Shape, n: int) -> float:
    """Attention matmul FLOPs (QK^T and AV) of one task through the current encoder.

    Args:
        s: Architecture.
        n: Rows per task.

    Returns:
        GFLOP across row-within-feature (``4 F N^2 d`` per layer), column (``4 N F^2 d``) and
        row (``4 N^2 D``) stages.
    """
    k = s.stage_layers
    rwf = 4.0 * s.features * n * n * s.d_cell * k
    col = 4.0 * n * s.features**2 * s.d_cell * k
    row = 4.0 * n * n * s.d_model * s.n_layers
    return (rwf + col + row) / 1e9


def attention_gflop_factorized(s: Shape, n: int, inducing: int = 64) -> float:
    """Attention matmul FLOPs of one task through a three-stage factorization.

    Stage 1 is column attention through ``inducing`` points (gather ``k x N`` then broadcast
    ``N x k``), stage 2 attends across features within a row, stage 3 attends across rows.

    Args:
        s: Architecture.
        n: Rows per task.
        inducing: Inducing points per feature, ``k``.

    Returns:
        GFLOP, with the same layer counts as the current encoder.
    """
    k = s.stage_layers
    stage1 = 4.0 * s.features * (2 * inducing * n) * s.d_cell * k
    stage2 = 4.0 * n * s.features**2 * s.d_cell * k
    stage3 = 4.0 * n * n * s.d_model * s.n_layers
    return (stage1 + stage2 + stage3) / 1e9


def linear_gflop(s: Shape, n: int) -> float:
    """Projection and feed-forward FLOPs of one task, identical in both encoders.

    Args:
        s: Architecture.
        n: Rows per task.

    Returns:
        GFLOP: ``16 d^2`` per token per layer (``d_ff = 2d`` in the cell stages) over
        ``2 K n_col_layers`` cell layers on ``F N`` tokens, plus ``8 D^2 + 4 D d_ff`` per row
        token per row layer with ``d_ff = 4 D`` as in the 5M model.
    """
    cell = 16.0 * s.d_cell**2 * s.features * n * 2 * s.stage_layers
    row = (8.0 * s.d_model**2 + 16.0 * s.d_model**2) * n * s.n_layers
    return (cell + row) / 1e9


def table_cpu() -> None:
    """Print table 1: CPU no-grad retrodiction against §79 and §81."""
    # (source, N, feature_chunk, measured GB, label)
    rows = [
        ("S79 cpu", 516, None, 2.1, "MEASURED"),
        ("S79 cpu", 1016, None, 6.7, "MEASURED"),
        ("S79 cpu", 2016, None, 15.9, "MEASURED"),
        ("S81 cpu", 2024, None, 21.0, "MEASURED"),
        ("S81 cpu", 2512, None, 27.2, "MEASURED"),
        ("S81 cpu", 2024, 16, 3.4, "MEASURED"),
        ("S81 cpu", 2512, 16, 6.5, "MEASURED"),
        ("S81 cpu", 3024, 16, 9.1, "MEASURED"),
        ("S81 cpu", 5024, 16, 15.7, "MEASURED"),
        ("S81 cpu", 6048, 16, 22.5, "MEASURED"),
        ("S79 est", 4048, None, 63.0, "ESTIMATED"),
    ]
    print("Table 1. CPU no-grad inference, 885K shape, F=136, B=1 (GB, peak resident)")
    print(f"{'source':8} {'N':>5} {'chunk':>5} {'low':>6} {'central':>7} {'high':>6} "
          f"{'measured':>8} {'ratio':>6}  within 2x  label")
    for src, n, chunk, meas, label in rows:
        lo, mid, hi = (cpu_peak_gb(SHAPE_885K, n, chunk, c) for c in CPU_SCORE_COPIES)
        inside = "yes" if 0.5 <= mid / meas <= 2.0 else "NO"
        print(f"{src:8} {n:>5} {chunk!s:>5} {lo:6.1f} {mid:7.1f} {hi:6.1f} "
              f"{meas:8.1f} {mid / meas:6.2f}  {inside:9}  {label}")
    print()


def table_cuda() -> None:
    """Print table 2: CUDA training retrodiction against §94 and §95."""
    print("Table 2. CUDA training (fp32, fused attention), saved-for-backward bytes (GB)")
    print(f"{'source':10} {'model':>5} {'B':>2} {'N':>5} {'saved':>7} {'measured':>9} "
          f"{'saved/meas':>10}  note")
    for n, meas in [(128, 0.85), (256, 1.67), (384, 2.49), (512, 3.31), (768, 4.96)]:
        saved = cuda_saved_gb(SHAPE_885K, 4, n, UNITS_885K)
        print(f"{'S95 L4':10} {'885K':>5} {4:>2} {n:>5} {saved:7.2f} {meas:9.2f} "
              f"{saved / meas:10.2f}  MEASURED peak allocated")
    saved512 = cuda_saved_gb(SHAPE_5M, 8, 512, UNITS_5M)
    overhead = 12.28 / saved512
    for n, meas in [(512, 12.28), (1024, None)]:
        saved = cuda_saved_gb(SHAPE_5M, 8, n, UNITS_5M)
        if meas is not None:
            print(f"{'S94 L4':10} {'5M':>5} {8:>2} {n:>5} {saved:7.2f} {meas:9.2f} "
                  f"{saved / meas:10.2f}  MEASURED, fits")
        else:
            peak = saved * overhead
            verdict = "OOM" if peak > 23.7 else "fits"
            print(f"{'S94 L4':10} {'5M':>5} {8:>2} {n:>5} {saved:7.2f} {'OOM':>9} "
                  f"{'':>10}  saved alone {saved / 23.7:.0%} of 23.7 GB; "
                  f"x{overhead:.3f} overhead from N=512 gives {peak:.1f} GB -> {verdict}")
    print()


def table_projection() -> None:
    """Print table 3: projection at V4FinBench's shape, current against three-stage."""
    s = SHAPE_5M
    print("Table 3. Projection, 5M shape, F=136, per task (B=1), ESTIMATED")
    print(f"{'N':>5} | {'score GB/layer (cur)':>20} {'(3-stage)':>10} | "
          f"{'GFLOP cur':>10} {'3-stage':>9} {'x':>5} | {'CUDA saved GB':>13} {'3-stage':>8}")
    for n in (512, 1024, 2024, 4096):
        cur_score = score_bytes_row_within_feature(s, n) / 1e9
        fac_score = s.features * s.heads * (2 * 64 * n) * FP32 / 1e9
        cur = attention_gflop_current(s, n) + linear_gflop(s, n)
        fac = attention_gflop_factorized(s, n) + linear_gflop(s, n)
        saved = cuda_saved_gb(s, 1, n, UNITS_5M)
        saved_fac = cuda_saved_gb(s, 1, n, UNITS_5M_INDUCED)
        print(f"{n:>5} | {cur_score:20.2f} {fac_score:10.3f} | {cur:10.0f} {fac:9.0f} "
              f"{cur / fac:5.1f} | {saved:13.2f} {saved_fac:8.2f}")
    print()


def main() -> None:
    """Print all three tables."""
    table_cpu()
    table_cuda()
    table_projection()


if __name__ == "__main__":
    main()
