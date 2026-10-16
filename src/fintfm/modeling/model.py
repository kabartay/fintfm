"""Prior-fitted network for tabular in-context classification.

Architecture, and why it is shaped this way
-------------------------------------------
A table has no canonical column order: permuting two features does not change what the data
means. A model that feeds a flat row vector through a single ``Linear`` gives every feature a
fixed weight slot and therefore *does* change its answer under permutation, cannot accept a
table narrower than the slot it was trained for without the padding position mattering, and
shares weights between column 7 of one dataset and column 7 of an unrelated one. That was
this file's first design and it was the main limit on transfer.

The current design is three stages, permutation-equivariant over columns and permutation-
invariant where it must be:

1. **Cell embedding.** Every cell becomes a token from its normalised value and a missingness
   flag. No feature-index embedding is added, deliberately: column identity must come from
   the data distribution, not from position.
2. **Column attention.** Cells within a row attend to each other, so a feature is
   contextualised by the other features of the same company. Padded columns are masked out.
   This stage is *equivariant*: permuting columns permutes the outputs identically.
3. **Row compression, then row attention.** Cells are pooled into one vector per row
   (*invariant* to column order), context rows receive their label embedding, and rows attend
   to context rows to perform in-context learning.

This is an independent implementation of the general alternating row/column attention idea
described in the public TabPFN / TabICL / TabFM literature (see ``docs/research/REFERENCES.md``). No
code or weights from those projects are used; see the licensing boundary in ``CLAUDE.md``.

Checkpoints written by the earlier flat-vector architecture are not loadable here. They were
smoke tests only and none were kept.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from fintfm.modeling.hazard import CENSORED, HazardHead

# Seed used for column identities whenever a caller does not supply one and the model is in
# eval mode. Its value is arbitrary; that it is *fixed* is the point (see _column_ids).
_DEFAULT_COLUMN_ID_SEED = 0

#: Accepted values of :attr:`ModelConfig.attn_impl`.
ATTN_IMPLS: tuple[str, ...] = ("masked", "split")

#: What :meth:`FinancialTFM.predict_context` returns for the query rows.
ContextOutput = Literal["logits", "term_structure", "representation"]
#: Depth :meth:`FinancialTFM.predict_context` reads a representation at.
RepresentationStage = Literal["encode_rows", "encoder_mid", "pre_head"]


@dataclass
class ModelConfig:
    """Architecture hyper-parameters.

    Attributes:
        max_features: Widest table the encoder accepts. Narrower tables are padded and the
            padding is masked out, so the padded width does not affect the answer.
        max_classes: Width of the classification head.
        d_cell: Width of the per-cell token in the column-attention stage. Kept below
            ``d_model`` because this stage runs on ``B * N`` sequences and dominates cost.
        d_model: Width of the row representation and the row-attention stage.
        n_heads: Attention heads, shared by both stages.
        n_col_layers: Column-attention layers. Two is usually enough to mix features.
        n_layers: Row-attention layers, where the in-context learning happens.
        d_ff: Feed-forward width in the row stage.
        dropout: Dropout probability. Zero for pretraining, where data is effectively
            infinite and there is nothing to overfit.
        pooling: How feature tokens are reduced to one row vector. ``"meanmax"`` is the
            original masked mean-and-max. ``"attention"`` uses a learned query attending over
            the feature tokens, so a feature's contribution can be *weighted* instead of
            averaged.

            **This is the one architecture change in the project with evidence behind it**
            (``docs/results/FINDINGS.md`` §50). A mean is weight-blind: every feature token
            contributes ``1/F`` however much it matters. Measured, the model scores 0.826 on a
            five-feature linear task and 0.551 at eighty — and three checkpoints spanning 17×
            in parameters produce *identical* curves to three decimal places, which rules out
            capacity and indicts the reduction itself.
        column_id_dim: Width of the random vector giving each column an identity. ``None``
            means ``d_cell // 4`` (TabPFN's ratio); ``0`` disables the mechanism and
            reproduces the pre-§54 architecture.

            **Without this the model cannot represent "column j matters."** ``cell_embed`` is
            shared across columns, the column encoder has no positional encoding, and pooling
            reduces over the feature axis — so the row representation is a *symmetric function
            of the multiset of that row's values* (``docs/results/FINDINGS.md`` §54, verified: the
            embedding converges to exactly permutation-invariant as the context grows). A rule
            like ``x_0 - x_1`` is then unlearnable in principle, and was measured at AUC 0.5097
            against a provable symmetric ceiling of 0.5.

            The fix is TabPFN's: draw a random vector per column, project it through a learned
            linear layer, add it to every cell of that column, and **resample it for every
            task**. Rows then agree on which column is which, while the distribution over
            tasks stays exactly column-order invariant — so D4 is preserved. Note this is
            needed *in addition to* column attention, not instead of it: equivariant attention
            alone still cannot separate two identically-distributed features.

            Because tags are resampled per forward pass, a single forward is stochastic. Use
            ``n_ensemble > 1`` at inference to average over draws, or seed ``torch`` for
            reproducibility.
        n_horizons: When set, the model also carries a :class:`HazardHead` producing a
            **provably monotone** cumulative-PD term structure over this many periods. The
            object IFRS 9 lifetime expected credit loss consumes, and the fix for the 39%
            incoherence measured in ``docs/results/FINDINGS.md`` §11. ``None`` keeps the model
            classification-only.
        n_cell_blocks: Number of alternating two-way cell-attention blocks run **before**
            pooling. ``0`` (default) reproduces every checkpoint trained before this field
            existed, byte-for-byte -- this is additive, not a replacement.

            **Why this exists.** ``docs/results/FINDINGS.md`` §74 found training on the financial
            prior caps basic signal extraction at ~0.73 AUC regardless of true task
            difficulty, on a probe with *no column-identity structure at all* -- one
            informative dimension, five inert companions. §76 bisected four content-side
            causes and none closed the gap. What the architecture has never had is a way for
            a cell to attend to *the rest of its own column* -- ``encode_rows``'s column
            stage "runs on one row at a time and so never sees a column" (see
            ``column_id_dim``'s docstring). Random column identities (fixed via
            ``column_id_dim``) let a cell know *which* column it is; they give it no way to
            learn what that column's values, across the whole context, actually look like.

            Each block alternates: attention **across rows within one feature** (new -- a
            cell attends to the same feature's cells in every other row, a data-derived
            column identity computed from the column's own distribution) then attention
            **across features within one row** (existing ``column_encoder``, reused). The
            row-within-feature stage reuses :meth:`_row_mask`, since which rows may attend to
            which does not depend on which feature is being processed.

            This is `openspec/changes/cell-attention-and-task-inference` task 39.1, and it is
            an experiment, not a presumed fix -- report the §74 probe's result on the
            resulting checkpoint before claiming it changed anything (task 39.4).
        cell_labels: When ``n_cell_blocks > 0``, inject the label into every cell of a
            context row *before* the cell-attention blocks (task 39.2), rather than only
            after pooling as the row-level ``y_proj`` step already does. Context cells get
            their true label broadcast across every feature; query cells get a learned mask
            token. The existing post-pooling injection is unchanged and still runs regardless
            -- this is additive, so a checkpoint trained with ``cell_labels=False`` differs
            from one trained without any label-throughout mechanism only in this one addition.
            Ignored when ``n_cell_blocks == 0``.
        mask_embedding: When ``True``, replace the default joint ``[value, missing]`` cell
            embedding with a dedicated missingness embedding (task 48.10) added to a
            value-only embedding: ``value_embed(Z) + mask_embed(missing)``, where
            ``mask_embed`` is an ``nn.Embedding(2, d_cell)`` rather than a second input
            channel to the same small MLP. ``False`` (default) reproduces every checkpoint
            trained before this field existed, byte for byte. Task 48.10's premise correction
            found the existing joint embedding is not naive imputation -- it already learns a
            value/missingness interaction through two linear layers and a GELU -- so this flag
            tests the narrower remaining question of whether a *dedicated* embedding table
            does better than that joint MLP, not whether missingness is encoded at all.
        head_type: ``"binned"`` (default) reproduces every checkpoint trained before this
            field existed, byte for byte: the classification head doubles as a regression
            head by having the caller bin a continuous target into ``max_classes`` quantile
            bins first (``fintfm.inference.binning``), so resolution is bounded by the bin
            count and an interval can never be narrower than one bin (``docs/results/FINDINGS.md``
            §106). ``"quantile"`` (task 48.3) replaces this with a genuine pinball-loss head:
            ``n_quantiles`` logit-free outputs, each trained to approximate one quantile
            level of the (context-normalised) continuous target directly, with no binning
            and no bound on resolution other than ``n_quantiles`` itself. Label injection
            (``cell_y_proj``, ``y_proj``) switches from a one-hot class vector to a single
            continuous scalar to match -- a model with ``head_type="quantile"`` only accepts
            continuous ``y`` and is not a classifier. Incompatible with ``n_horizons`` (the
            hazard head assumes a discrete class head beneath it; no survival-plus-quantile
            checkpoint has been designed).
        n_quantiles: Output width of a ``"quantile"`` head -- the number of quantile levels
            predicted per row, evenly spaced in ``(0, 1)``. Nori's published head uses 999;
            this project starts an order of magnitude smaller; it is not tuned. Ignored when
            ``head_type == "binned"``.

        **Audit Phase 5 experiment levers** (``openspec/changes/audit-2026-10-15``). Every
        field below defaults to the architecture that existed before it, byte for byte, and a
        checkpoint whose saved config lacks these keys loads with exactly those defaults. None
        has a measured effect yet; each is an arm to score, not an improvement.

        n_cell_heads: Attention heads in the cell-width stages (column attention, row-within-
            feature attention, attention pooling and summary tokens). ``None`` (default)
            shares ``n_heads``, as every checkpoint before this field did. Separate because
            ``d_cell`` is usually much narrower than ``d_model``, so one head count rarely
            suits both widths (AU5.3).
        n_summary_tokens: Learned summary tokens per row (AU5.3). ``1`` (default) keeps the
            single pooled row vector of ``pooling``. ``K > 1`` replaces that pooling with
            ``K`` learned queries that cross-attend over the row's feature tokens; their
            outputs are concatenated (``K * d_cell``) and projected to ``d_model``, so the
            row stage is unchanged. Cross-attention over a key set is order-invariant, so
            column-order invariance (D4) holds. Unlike TabICL's CLS tokens these do not take
            part in column attention, which keeps the change inside the pooling step.
            Incompatible with ``pooling="attention"``, which is the ``K = 1`` case of the
            same idea plus mean and max.
        value_embedding: How a normalised cell value enters the cell embedding (AU5.16).
            ``"mlp"`` (default) is the existing small MLP on the scalar. ``"rbf"`` adds
            ``value_basis_size`` Gaussian bumps over the normalised value; ``"fourier"`` adds
            ``value_basis_size`` random-frequency sine/cosine pairs. Both keep the raw value
            as a channel beside the basis and feed the same two-layer MLP, so the flag
            changes the input features and nothing downstream. See :class:`ValueBasis`.
        value_basis_size: Number of RBF centres, or of Fourier frequencies (each giving a
            sine and a cosine). Ignored for ``value_embedding="mlp"``.
        value_basis_learnable: Train the RBF centres and widths, or the Fourier frequencies,
            instead of keeping them at their initial values. Ignored for ``"mlp"``.
        value_ecdf: Add a channel holding each cell's empirical CDF rank within its column,
            computed against the **context rows only** (:func:`ecdf_rank`), so a query value
            never affects any other row's features. Combines with any ``value_embedding``.
        attn_length_scale: Length-aware softmax scaling in the row (in-context) attention
            (AU5.17, the attention half of TabICLv2's QASSMax idea, without its learned
            per-query factor). The attention logits are multiplied by
            ``log(n_ctx) / log(attn_length_ref)``; see :func:`length_scale` for the formula
            and why. ``False`` (default) runs the stock encoder. The cell-stage attentions
            are not scaled.
        attn_length_ref: Context length at which the scaled attention equals the stock one.
            Set near the pretraining context so training is a small perturbation of the
            unscaled model there; longer inference contexts then get sharper logits rather
            than flatter attention. Ignored unless ``attn_length_scale``.
        attn_impl: How the context-plus-self row attention is computed (AU5.2), in the row
            encoder and, with ``n_cell_blocks > 0``, the row-within-feature stage. Both give
            the same outputs to float tolerance and use the same weights, so a checkpoint
            trained under either loads and runs under either. ``"masked"`` (default) is the
            stock encoder with :meth:`FinancialTFM._row_mask` as a boolean mask, which no
            fused kernel accepts. ``"split"`` (:func:`split_row_attention`) runs the context
            rows' attention with no mask at all and the query rows' attention over the
            context keys plus one appended key that carries each query's own key and value,
            again with no mask, so both calls are eligible for CUDA's flash and
            memory-efficient kernels. The speed difference is a property of the device and
            must be measured there.
        hazard_period_context: Context rows of a hazard model carry their default period, or
            a censored token, in the label embedding as well as the binary label (AU5.9).
            Without it the context says only *whether* a firm defaulted, so the term
            structure cannot be read from context in principle. Used by
            :meth:`term_structure` and :meth:`survival_loss` only, which then require
            ``period``; :meth:`forward` (the classification path) is unchanged. Requires
            ``n_horizons``.
    """

    max_features: int = 24
    max_classes: int = 10
    d_cell: int = 64
    d_model: int = 192
    n_heads: int = 4
    n_col_layers: int = 2
    n_layers: int = 6
    d_ff: int = 512
    dropout: float = 0.0
    pooling: str = "meanmax"
    column_id_dim: int | None = None
    n_horizons: int | None = None
    n_cell_blocks: int = 0
    cell_labels: bool = False
    mask_embedding: bool = False
    head_type: str = "binned"
    n_quantiles: int = 99
    n_cell_heads: int | None = None
    n_summary_tokens: int = 1
    value_embedding: str = "mlp"
    value_basis_size: int = 16
    value_basis_learnable: bool = False
    value_ecdf: bool = False
    attn_length_scale: bool = False
    attn_length_ref: int = 512
    attn_impl: str = "masked"
    hazard_period_context: bool = False

    def __post_init__(self) -> None:
        """Validate ``head_type``, the Phase 5 levers, and their interactions.

        Raises:
            ValueError: If ``head_type`` is not one of the two supported values, or if a
                quantile head is combined with a hazard head -- a combination nothing in this
                codebase builds, scores, or has a designed loss for. Also for an unknown
                ``value_embedding``, a non-positive size, summary tokens combined with
                attention pooling, ``hazard_period_context`` without a hazard head, or an
                unknown ``attn_impl``.
        """
        if self.value_embedding not in ("mlp", "rbf", "fourier"):
            raise ValueError(
                "value_embedding must be 'mlp', 'rbf' or 'fourier', "
                f"got {self.value_embedding!r}"
            )
        if self.value_basis_size < 1:
            raise ValueError(f"value_basis_size must be >= 1, got {self.value_basis_size}")
        if self.n_summary_tokens < 1:
            raise ValueError(f"n_summary_tokens must be >= 1, got {self.n_summary_tokens}")
        if self.n_summary_tokens > 1 and self.pooling == "attention":
            raise ValueError(
                "n_summary_tokens > 1 replaces the pooling step, so it cannot be combined "
                "with pooling='attention'"
            )
        if self.n_cell_heads is not None and self.n_cell_heads < 1:
            raise ValueError(f"n_cell_heads must be >= 1, got {self.n_cell_heads}")
        if self.attn_length_ref < 2:
            raise ValueError(f"attn_length_ref must be >= 2, got {self.attn_length_ref}")
        if self.attn_impl not in ATTN_IMPLS:
            raise ValueError(f"attn_impl must be one of {ATTN_IMPLS}, got {self.attn_impl!r}")
        if self.hazard_period_context and self.n_horizons is None:
            raise ValueError("hazard_period_context requires n_horizons (a hazard head)")
        if self.head_type not in ("binned", "quantile"):
            raise ValueError(
                f"head_type must be 'binned' or 'quantile', got {self.head_type!r}"
            )
        if self.head_type == "quantile" and self.n_horizons is not None:
            raise ValueError(
                "head_type='quantile' is incompatible with n_horizons: the hazard head "
                "assumes a discrete class head beneath it, and no survival-plus-quantile "
                "design exists"
            )


def normalize_features(
    X: torch.Tensor, n_ctx: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Z-score features per task using *context* statistics only.

    Using context rows alone is what keeps query rows from leaking into their own
    normalisation, which would be a subtle form of test-set contamination.

    Args:
        X: ``(B, N, F)`` raw values, NaN for both missing cells and padded columns.
        n_ctx: Number of leading context rows whose statistics are used.

    Returns:
        A tuple of:
            - ``Z``: ``(B, N, F)`` normalised values, clipped, with missing cells set to 0.
            - ``missing``: ``(B, N, F)`` float mask, 1.0 where the cell was NaN.
            - ``pad``: ``(B, F)`` bool mask, True for columns that are entirely absent
              from the *context* and are therefore padding rather than data.
    """
    ctx = X[:, :n_ctx]
    present = ~torch.isnan(ctx)
    cnt = present.sum(dim=1, keepdim=True).clamp(min=1)
    mean = torch.nan_to_num(ctx).sum(dim=1, keepdim=True) / cnt
    var = (torch.nan_to_num(ctx - mean) ** 2 * present).sum(dim=1, keepdim=True) / cnt
    std = torch.sqrt(var + 1e-6)
    Z = (X - mean) / std
    missing = torch.isnan(Z)
    Z = torch.nan_to_num(Z).clamp(-10.0, 10.0)
    Z = torch.where(missing, torch.zeros_like(Z), Z)
    # Context rows only, like the statistics above. Reading query rows here let one query
    # with a value in a context-empty column un-pad that column for every row of the task,
    # so a query's output depended on which other queries shared its forward pass and
    # chunked prediction stopped being exact (audit AU1.1). A context-empty column carries
    # no evidence to compare a query against, so treating it as padding loses nothing.
    pad = torch.isnan(ctx).all(dim=1)
    return Z, missing.to(Z.dtype), pad


def pack_context(
    Xc: np.ndarray,
    yc: np.ndarray,
    Xq: np.ndarray,
    width: int,
    y_dtype: type[np.generic],
    period: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Lay one context and its query rows out as a single task, the model's input shape.

    The one padding helper of DR1.4: every inference path used to copy these lines, and a
    change to one copy (a dtype, a placeholder) would have reached one path and missed the
    rest (``CLAUDE.md``, "a correction that lives in one method").

    Args:
        Xc: ``(n_ctx, f)`` context rows.
        yc: ``(n_ctx,)`` context labels.
        Xq: ``(n_q, f)`` query rows.
        width: Feature axis of the task, ``>= f``; columns past ``f`` are NaN, which the
            model reads as padding.
        y_dtype: Label dtype: ``np.int64`` class indices, or ``np.float32`` for a quantile
            head's continuous target.
        period: Optional ``(n_ctx,)`` context default periods.

    Returns:
        ``(X, y, period)`` of shapes ``(1, N, width)`` float32, ``(1, N)`` and ``(1, N)``
        int64 (or ``None`` without ``period``), ``N = n_ctx + n_q``. Query labels are 0 and
        query periods ``CENSORED``: placeholders the model never reads, since only the first
        ``n_ctx`` labels and periods are inputs.
    """
    n_ctx, n = Xc.shape[0], Xc.shape[0] + Xq.shape[0]
    X = np.full((1, n, width), np.nan, dtype=np.float32)
    X[0, :n_ctx, : Xc.shape[1]] = Xc
    X[0, n_ctx:, : Xq.shape[1]] = Xq
    y = np.zeros((1, n), dtype=y_dtype)
    y[0, :n_ctx] = yc
    if period is None:
        return X, y, None
    pp = np.full((1, n), CENSORED, dtype=np.int64)
    pp[0, :n_ctx] = period
    return X, y, pp


def ecdf_rank(X: torch.Tensor, n_ctx: int) -> torch.Tensor:
    """Each cell's mid-rank empirical CDF within its column, against context rows only.

    The reference set is the column's present **context** values, never the query rows, for
    the same reason :func:`normalize_features` uses context statistics: a query value that
    entered the reference would shift every other row's feature, so a prediction would
    depend on which other queries shared its forward pass (audit AU1.1's failure shape).
    Ties count half (mid-rank), so a value equal to every context value maps to the centre.

    Computed by sorting the context once per column and binary-searching every cell, so the
    cost is ``O(N log n_ctx)`` per column rather than the ``O(N * n_ctx)`` of a pairwise
    comparison, which at ``N = 4096`` and 136 features would not fit in memory.

    Args:
        X: ``(B, N, F)`` raw values, NaN for missing cells and padded columns.
        n_ctx: Number of leading context rows forming the reference set.

    Returns:
        ``(B, N, F)`` ranks mapped from ``[0, 1]`` to ``[-1, 1]``, so the column median sits
        at 0. Missing cells and columns with no context values are 0.
    """
    ctx = X[:, :n_ctx].transpose(1, 2)  # (B, F, n_ctx)
    present = ~torch.isnan(ctx)
    n_present = present.sum(dim=-1, keepdim=True)  # (B, F, 1)
    # Missing context cells become +inf, so they sort after every finite value and are
    # never counted as below or equal to a finite query value.
    ref = torch.where(present, ctx, torch.full_like(ctx, float("inf"))).sort(dim=-1).values
    vals = X.transpose(1, 2).contiguous()  # (B, F, N)
    absent = torch.isnan(vals)
    probe = torch.where(absent, torch.zeros_like(vals), vals)
    below = torch.searchsorted(ref.contiguous(), probe, side="left")
    at_or_below = torch.searchsorted(ref.contiguous(), probe, side="right")
    u = (below + at_or_below).to(X.dtype) / (2.0 * n_present.clamp(min=1).to(X.dtype))
    r = (2.0 * u.clamp(0.0, 1.0) - 1.0).masked_fill(absent | (n_present == 0), 0.0)
    return r.transpose(1, 2)


def length_scale(n_keys: int, n_ref: int) -> float:
    """Multiplier on the row-attention logits for a context of ``n_keys`` rows (AU5.17).

    ``factor = log(max(n_keys, 2)) / log(n_ref)``, applied on top of the usual
    ``1 / sqrt(d_head)``, so the logits are ``factor * q.k / sqrt(d_head)``.

    Why this form. With logits of bounded spread, softmax over ``n`` keys gives the largest
    weight about ``1 / n`` of the mass once ``n`` dominates the exponentiated spread, so the
    same learned query attends more diffusely the longer the context: a model pretrained at
    a few hundred rows and served at 4,096 sees flatter attention than it ever trained on.
    Multiplying the logits by ``log n`` (scalable softmax) keeps a logit gap of fixed size
    worth a fixed share of the mass as ``n`` grows. Dividing by ``log(n_ref)`` makes the
    factor exactly 1 at the reference length, so near the pretraining context the scaled
    model is the stock model, and the flag only changes behaviour away from it. ``n_keys``
    is the context-row count: the quantity that varies between tasks. A query row also sees
    itself, one key more, which is ignored. The floor at 2 keeps the factor positive; a
    one-row context is degenerate anyway.

    TabICLv2 learns a per-query multiplier on top of ``log n`` (QASSMax); that part is not
    implemented here, so this is the fixed-scale variant only.

    Args:
        n_keys: Number of context rows the attention runs over.
        n_ref: Reference length at which the factor is 1. Must be at least 2.

    Returns:
        The multiplier.
    """
    return math.log(max(n_keys, 2)) / math.log(n_ref)


def split_row_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_ctx: int,
    scale: float,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Context-plus-self row attention without an attention mask (AU5.2).

    Computes what :func:`~torch.nn.functional.scaled_dot_product_attention` computes under
    :meth:`FinancialTFM._row_mask`: a context row attends to every context row, a query row
    to every context row and to itself, and to nothing else. That mask is what keeps CUDA's
    fused kernels out, since the flash kernel takes no mask at all. Here neither call has one:

    * **Context rows:** plain self-attention over the first ``n_ctx`` rows.
    * **Query rows:** one call against the ``n_ctx`` context keys plus a single appended
      *sentinel* key, after widening the head dimension by ``e`` channels (``e >= 1``,
      chosen so the width stays a multiple of 8, which the flash kernel requires). Every
      context key gets zeros in the new channels, the sentinel is zero except a 1 in the
      first new channel, and each query carries its own ``q_i . k_i`` there. So query ``i``'s
      logit against context row ``j`` is still ``q_i . k_j``, and against the sentinel it is
      ``q_i . k_i``: the self logit, per query, through a key shared by all of them. The
      values are widened the same way, the sentinel's value being the indicator of the first
      new channel, so that channel of the output is the softmax weight the query put on
      itself, ``p_self``, and the original channels hold the context part already weighted
      by ``1 - p_self``. Adding ``p_self * v_i`` completes the attention over
      ``[context, self]``. Every step is differentiable, so training uses it unchanged.

    Exact up to floating-point summation order: the softmax runs over the same logits in a
    different arrangement. Dropout, when on, drops the self weight with the same probability
    as any other attention weight, as the masked path does; the two draw different masks.

    Args:
        q: ``(B, H, N, d)`` queries, rows ordered context first.
        k: ``(B, H, N, d)`` keys.
        v: ``(B, H, N, d)`` values.
        n_ctx: Number of context rows, ``0 <= n_ctx <= N``.
        scale: Logit scale, ``1 / sqrt(d)`` for the stock attention.
        dropout_p: Attention dropout probability.

    Returns:
        ``(B, H, N, d)``.
    """
    d = q.shape[-1]
    parts = []
    if n_ctx > 0:
        k_ctx, v_ctx = k[..., :n_ctx, :], v[..., :n_ctx, :]
        parts.append(
            F.scaled_dot_product_attention(
                q[..., :n_ctx, :], k_ctx, v_ctx, dropout_p=dropout_p, scale=scale
            )
        )
    else:
        k_ctx, v_ctx = k[..., :0, :], v[..., :0, :]
    if n_ctx < q.shape[-2]:
        q_qry, k_qry, v_qry = q[..., n_ctx:, :], k[..., n_ctx:, :], v[..., n_ctx:, :]
        extra = 8 - d % 8  # at least one channel, and a width the flash kernel accepts
        self_logit = (q_qry * k_qry).sum(dim=-1, keepdim=True)
        q_aug = torch.cat([q_qry, F.pad(self_logit, (0, extra - 1))], dim=-1)
        sentinel = torch.zeros(*k.shape[:-2], 1, d + extra, dtype=k.dtype, device=k.device)
        sentinel[..., d] = 1.0
        k_aug = torch.cat([F.pad(k_ctx, (0, extra)), sentinel], dim=-2)
        v_aug = torch.cat([F.pad(v_ctx, (0, extra)), sentinel.to(v.dtype)], dim=-2)
        out = F.scaled_dot_product_attention(
            q_aug, k_aug, v_aug, dropout_p=dropout_p, scale=scale
        )
        parts.append(out[..., :d] + out[..., d : d + 1] * v_qry)
    return torch.cat(parts, dim=-2)


class ValueBasis(nn.Module):
    """Fixed-width basis expansion of a normalised cell value (AU5.16).

    A two-layer MLP on a single scalar can represent sharp or periodic responses to a value
    only by spending most of its hidden units on them. Expanding the scalar into a basis
    first hands the MLP those shapes directly: ``"rbf"`` gives Gaussian bumps at fixed
    positions (local, so a threshold anywhere in the bulk is one weight away), ``"fourier"``
    gives random-frequency sinusoids (global, the random-features approximation of the same
    Gaussian kernel). This is the general idea behind LimiX's RaBEL and TabPFN's value
    encodings as described in their public write-ups; it is an independent implementation.

    Both kinds use the same length scale, ``ell = 2 * RBF_HALF_RANGE / (size - 1)`` (the
    RBF spacing), so ``"rbf"`` and ``"fourier"`` approximate the same kernel and the
    comparison between them is about the parameterisation, not the resolution. Fourier
    frequencies are drawn as ``omega ~ N(0, 1 / ell**2)``, which is the spectral density of
    a Gaussian kernel of width ``ell``, from a private generator with a fixed seed, so the
    draw neither consumes nor depends on the global RNG. Both are stored in the checkpoint.

    Args:
        kind: ``"rbf"`` or ``"fourier"``.
        size: Number of centres, or of frequencies (each yields a sine and a cosine).
        learnable: Make centres and log-widths, or frequencies, trainable parameters.
    """

    #: RBF centres span ``[-3, 3]``: the bulk of a z-scored or rank-to-normal column. Values
    #: beyond it still reach the MLP through the raw-value channel kept beside the basis.
    RBF_HALF_RANGE = 3.0
    #: Seed of the private generator for Fourier frequencies. Arbitrary; fixed on purpose.
    FOURIER_SEED = 0

    def __init__(self, kind: str, size: int, learnable: bool) -> None:
        """Build the basis; see the class docstring for the arguments."""
        super().__init__()
        if kind not in ("rbf", "fourier"):
            raise ValueError(f"kind must be 'rbf' or 'fourier', got {kind!r}")
        self.kind = kind
        ell = 2.0 * self.RBF_HALF_RANGE / (size - 1) if size > 1 else 1.0
        if kind == "rbf":
            centres = torch.linspace(-self.RBF_HALF_RANGE, self.RBF_HALF_RANGE, size)
            log_width = torch.full((size,), math.log(ell))
            if learnable:
                self.centres = nn.Parameter(centres)
                self.log_width = nn.Parameter(log_width)
            else:
                self.register_buffer("centres", centres)
                self.register_buffer("log_width", log_width)
            self.out_dim = size
        else:
            g = torch.Generator().manual_seed(self.FOURIER_SEED)
            freqs = torch.randn(size, generator=g) / ell
            if learnable:
                self.freqs = nn.Parameter(freqs)
            else:
                self.register_buffer("freqs", freqs)
            self.out_dim = 2 * size

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Expand values.

        Args:
            z: ``(...)`` normalised values.

        Returns:
            ``(..., out_dim)`` basis features, each in ``[-1, 1]``.
        """
        z = z.unsqueeze(-1)
        if self.kind == "rbf":
            centres = self.centres.to(z.dtype)
            width = self.log_width.to(z.dtype).exp()
            return torch.exp(-0.5 * ((z - centres) / width) ** 2)
        angle = z * self.freqs.to(z.dtype)
        return torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)


class FinancialTFM(nn.Module):
    """Tabular in-context classifier.

    Args:
        cfg: :class:`ModelConfig`.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        """Build the network from a configuration.

        Args:
            cfg: Architecture configuration. Every field is load-bearing at load time too —
                it is stored in the checkpoint and used to reconstruct the model, so a field
                added later needs a default in :meth:`load` or old checkpoints stop opening.
        """
        super().__init__()
        self.cfg = cfg
        # Runtime-only, deliberately NOT a ModelConfig field: it changes no weight and no
        # output, so it must not enter a checkpoint or require a load-time backfill. See
        # :attr:`feature_chunk`.
        self.feature_chunk: int | None = None
        # AU5.16: optional basis expansion and ECDF channel ahead of the cell MLP. With both
        # off the MLP's input is the scalar value alone, exactly as before these existed.
        self.value_basis = (
            ValueBasis(cfg.value_embedding, cfg.value_basis_size, cfg.value_basis_learnable)
            if cfg.value_embedding != "mlp"
            else None
        )
        n_value_inputs = (
            1
            + (self.value_basis.out_dim if self.value_basis is not None else 0)
            + int(cfg.value_ecdf)
        )
        if cfg.mask_embedding:
            # Dedicated missingness embedding (task 48.10), added to a value-only embedding
            # of matching depth rather than concatenated as a second input channel.
            self.value_embed = nn.Sequential(
                nn.Linear(n_value_inputs, cfg.d_cell),
                nn.GELU(),
                nn.Linear(cfg.d_cell, cfg.d_cell),
            )
            self.mask_embed = nn.Embedding(2, cfg.d_cell)
        else:
            self.cell_embed = nn.Sequential(
                nn.Linear(n_value_inputs + 1, cfg.d_cell),
                nn.GELU(),
                nn.Linear(cfg.d_cell, cfg.d_cell),
            )
        # AU5.3: the cell-width stages may use their own head count; None shares n_heads.
        cell_heads = cfg.n_heads if cfg.n_cell_heads is None else cfg.n_cell_heads
        # Random per-task column identities (§54). Resolved here so ``None`` can mean
        # "the sensible default" without the training entry point having to know d_cell.
        self.col_id_dim = (
            max(1, cfg.d_cell // 4) if cfg.column_id_dim is None else int(cfg.column_id_dim)
        )
        self.col_id_proj = (
            nn.Linear(self.col_id_dim, cfg.d_cell, bias=False) if self.col_id_dim > 0 else None
        )
        col_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_cell,
            nhead=cell_heads,
            dim_feedforward=2 * cfg.d_cell,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.column_encoder = nn.TransformerEncoder(
            col_layer, num_layers=cfg.n_col_layers, enable_nested_tensor=False
        )
        # Two-way cell attention (§74, §76, task 39.1). A separate stack from
        # column_encoder, same depth, so each block spends equal capacity attending across
        # rows-within-a-feature and across features-within-a-row. Built only when needed:
        # cfg.n_cell_blocks == 0 must leave the parameter count and every forward path
        # byte-identical to a pre-39.1 checkpoint.
        if cfg.n_cell_blocks > 0:
            row_within_feature_layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_cell,
                nhead=cell_heads,
                dim_feedforward=2 * cfg.d_cell,
                dropout=cfg.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.row_within_feature_encoder = nn.TransformerEncoder(
                row_within_feature_layer, num_layers=cfg.n_col_layers, enable_nested_tensor=False
            )
            if cfg.cell_labels:
                # Broadcast across every feature of a context row (task 39.2); query rows get
                # a learned mask token instead of a real label, matching the row-level
                # query_token pattern below. A quantile head's label is a single continuous
                # scalar rather than a one-hot class vector (task 48.3), so its projection
                # has in_features=1 instead of max_classes.
                label_width = 1 if cfg.head_type == "quantile" else cfg.max_classes
                self.cell_y_proj = nn.Linear(label_width, cfg.d_cell, bias=False)
                self.cell_mask_token = nn.Parameter(torch.zeros(cfg.d_cell))
        # Pooling over columns is a masked mean (order-invariant) plus a masked max, which
        # keeps a signal a mean washes out: one extreme ratio in an otherwise ordinary firm.
        if cfg.pooling not in ("meanmax", "attention"):
            raise ValueError(
                f"pooling must be 'meanmax' or 'attention', got {cfg.pooling!r}"
            )
        if cfg.n_summary_tokens > 1:
            # AU5.3: K learned queries cross-attend over a row's feature tokens and their
            # outputs are concatenated, replacing mean+max. Same small init as pool_query.
            self.summary_queries = nn.Parameter(
                torch.randn(1, cfg.n_summary_tokens, cfg.d_cell) * 0.02
            )
            self.summary_attn = nn.MultiheadAttention(
                cfg.d_cell, cell_heads, dropout=cfg.dropout, batch_first=True
            )
            self.row_proj = nn.Linear(cfg.n_summary_tokens * cfg.d_cell, cfg.d_model)
        elif cfg.pooling == "attention":
            # A single learned query attends over the feature tokens, so the reduction can
            # weight features instead of averaging them (§50). Mean and max are kept
            # alongside: they are cheap, they carry the summary statistics attention would
            # otherwise have to rediscover, and keeping them makes the change additive rather
            # than a replacement whose regressions would be hard to attribute.
            self.pool_query = nn.Parameter(torch.randn(1, 1, cfg.d_cell) * 0.02)
            self.pool_attn = nn.MultiheadAttention(
                cfg.d_cell, cell_heads, dropout=cfg.dropout, batch_first=True
            )
            self.row_proj = nn.Linear(3 * cfg.d_cell, cfg.d_model)
        else:
            self.row_proj = nn.Linear(2 * cfg.d_cell, cfg.d_model)
        # Same scalar-vs-one-hot split as cell_y_proj above, at the row level.
        self.y_proj = nn.Linear(
            1 if cfg.head_type == "quantile" else cfg.max_classes, cfg.d_model, bias=False
        )
        self.query_token = nn.Parameter(torch.zeros(cfg.d_model))
        row_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            row_layer, num_layers=cfg.n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        if cfg.head_type == "quantile":
            self.head = nn.Linear(cfg.d_model, cfg.n_quantiles)
            # Evenly spaced, strictly inside (0, 1): the two endpoints are undefined
            # quantiles (an infinite tail), so they are excluded rather than clamped.
            # Registered as a buffer so it moves with .to(device) and round-trips through
            # save/load without appearing in the optimiser's parameter list.
            self.register_buffer(
                "quantile_taus", torch.linspace(0.0, 1.0, cfg.n_quantiles + 2)[1:-1]
            )
        else:
            self.head = nn.Linear(cfg.d_model, cfg.max_classes)
        self.hazard = (
            HazardHead(cfg.d_model, cfg.n_horizons) if cfg.n_horizons is not None else None
        )
        if cfg.hazard_period_context:
            # AU5.9: index 0 is the censored token, index p + 1 a default in period p.
            # Zero-initialised so a fresh model computes exactly what it would without the
            # flag; the periods enter only as training moves this table off zero.
            assert cfg.n_horizons is not None  # checked in ModelConfig.__post_init__
            self.period_embed = nn.Embedding(cfg.n_horizons + 1, cfg.d_model)
            nn.init.zeros_(self.period_embed.weight)

    @staticmethod
    def _row_mask(n_rows: int, n_ctx: int, device: torch.device) -> torch.Tensor:
        """Boolean ``(N, N)`` row-attention mask, True = blocked.

        Every row may attend to context rows, and additionally to *itself*. Blocking the
        diagonal as well would leave a query row's own features reaching the output only
        through the residual path. Self-attention leaks nothing: a query row carries no
        label. Queries still never see each other, so a prediction never depends on which
        other rows happen to be in the same batch.
        """
        mask = torch.zeros(n_rows, n_rows, dtype=torch.bool, device=device)
        mask[:, n_ctx:] = True
        eye = torch.arange(n_rows, device=device)
        mask[eye, eye] = False
        return mask

    def _row_within_feature(
        self,
        by_feature: torch.Tensor,
        row_mask: torch.Tensor | None,
        B: int,
        Fdim: int,
        n_ctx: int | None = None,
    ) -> torch.Tensor:
        """Run row-attention-within-feature, optionally in chunks over the feature axis.

        The ``B * F`` attention problems stacked in ``by_feature``'s batch dimension are
        mutually independent -- a cell attends only to the same feature's cells in other
        rows -- and ``row_mask`` does not depend on which feature is being processed. So
        splitting the batch dimension into groups of whole features and concatenating the
        results is an identity, not an approximation.

        **Why it is worth doing.** This stage's attention scores are ``(B*F, heads, N, N)``,
        a feature-count factor the pooled architecture never carried. ``docs/results/FINDINGS.md``
        §79 measured ~16 GB at ``N=2024`` on V4FinBench's 136 features and a 92x performance
        cliff by ``N=2512``, which put §71's best inference configuration
        (``max_context=4000``, an estimated ~63 GB) out of reach entirely and forced §80 to
        compare two architectures at a context §31/§33 had measured the *older* one still
        improving past. Chunking trades that memory back for time and changes no number, so
        the comparison §80 could not make becomes possible.

        Args:
            by_feature: ``(B*F, N, d_cell)``, features varying slowest.
            row_mask: ``(N, N)`` boolean attention mask, True = blocked; ``None`` under
                ``attn_impl="split"``, which needs no mask.
            B: Batch size, used only to validate the chunking arithmetic.
            Fdim: Feature count, used only to validate the chunking arithmetic.
            n_ctx: Context/query split; read only when ``row_mask`` is ``None``.

        Returns:
            ``(B*F, N, d_cell)``, the encoder applied to every feature.
        """
        def encode(x: torch.Tensor) -> torch.Tensor:
            if row_mask is not None:
                return self.row_within_feature_encoder(x, mask=row_mask)
            assert n_ctx is not None
            return self._prenorm_stack(
                self.row_within_feature_encoder.layers, x, n_ctx, None, 1.0
            )

        chunk = self.feature_chunk
        if chunk is None or chunk >= Fdim:
            return encode(by_feature)
        if chunk < 1:
            raise ValueError(f"feature_chunk must be >= 1, got {chunk}")
        # Chunk in whole features so each group stays a set of independent problems. The
        # batch dimension is (B, F) with F varying fastest, so a stride of `chunk` features
        # is not contiguous across B -- reshape to (B, F, ...) and slice the feature axis
        # rather than slicing the flattened batch, which would mix features across tasks.
        view = by_feature.reshape(B, Fdim, *by_feature.shape[1:])
        out = torch.empty_like(view)
        for start in range(0, Fdim, chunk):
            stop = min(start + chunk, Fdim)
            group = view[:, start:stop].reshape(-1, *by_feature.shape[1:])
            out[:, start:stop] = encode(group).reshape(B, stop - start, *by_feature.shape[1:])
        return out.reshape_as(by_feature)

    def _column_ids(
        self, B: int, Fdim: int, device: torch.device, dtype: torch.dtype, seed: int | None
    ) -> torch.Tensor:
        """Draw the random per-task column identity vectors.

        **Random while training, deterministic while evaluating.** Training wants a fresh
        draw every step, because it is the resampling that stops any column index acquiring a
        fixed meaning and so keeps the task distribution column-order invariant. Inference
        wants reproducibility: a credit model whose score changes between two identical calls
        fails model validation before anyone looks at its accuracy, and stochastic predictions
        are exactly what ``docs/results/FINDINGS.md`` §55 argues supervisors grade against.

        Generated on the CPU and moved, so the stream does not depend on the accelerator and
        a seeded prediction reproduces across CPU, Metal and CUDA alike.
        """
        if seed is None and self.training:
            ids = torch.randn(B, Fdim, self.col_id_dim, dtype=torch.float32)
        else:
            g = torch.Generator().manual_seed(_DEFAULT_COLUMN_ID_SEED if seed is None else seed)
            ids = torch.randn(B, Fdim, self.col_id_dim, generator=g, dtype=torch.float32)
        return ids.to(device=device, dtype=dtype)

    def encode_rows(
        self,
        X: torch.Tensor,
        n_ctx: int,
        column_id_seed: int | None = None,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Turn raw cells into one vector per row.

        Args:
            X: ``(B, N, F)`` raw features.
            n_ctx: Context/query split, used for normalisation statistics.
            column_id_seed: Seed for the random column identities. ``None`` means a fresh
                draw while training and a fixed one while evaluating; pass distinct integers
                to ensemble over draws.
            y: ``(B, N)`` labels, only the first ``n_ctx`` read. Required when
                ``cfg.n_cell_blocks > 0 and cfg.cell_labels``; ignored otherwise, so every
                existing caller that does not pass it keeps working unchanged.

        Returns:
            ``(B, N, d_model)`` row representations.

        Raises:
            ValueError: If ``X`` is wider than ``cfg.max_features`` (design review DR1.5).
                Nothing in the weights depends on the width, so a wider table used to run
                and return numbers the model was never built or trained to produce.
        """
        B, N, Fdim = X.shape
        if Fdim > self.cfg.max_features:
            raise ValueError(
                f"the input table is {Fdim} features wide and ModelConfig.max_features is "
                f"{self.cfg.max_features}; the model never saw a wider table, so it refuses "
                "one rather than scoring it"
            )
        Z, missing, pad = normalize_features(X, n_ctx)
        if self.value_basis is None and not self.cfg.value_ecdf:
            # The original value channel, kept as its own branch so the default path is the
            # pre-AU5.16 code byte for byte.
            if self.cfg.mask_embedding:
                cells = self.value_embed(Z.unsqueeze(-1)) + self.mask_embed(missing.long())
            else:
                cells = self.cell_embed(torch.stack([Z, missing], dim=-1))  # (B, N, F, d_cell)
        else:
            # AU5.16: raw value, then the basis (zeroed on missing cells so a missing value
            # is not read as a value of 0), then the context-only ECDF rank.
            value = [Z.unsqueeze(-1)]
            if self.value_basis is not None:
                value.append(self.value_basis(Z) * (1.0 - missing).unsqueeze(-1))
            if self.cfg.value_ecdf:
                value.append(ecdf_rank(X, n_ctx).to(Z.dtype).unsqueeze(-1))
            v = torch.cat(value, dim=-1)
            if self.cfg.mask_embedding:
                cells = self.value_embed(v) + self.mask_embed(missing.long())
            else:
                cells = self.cell_embed(torch.cat([v, missing.unsqueeze(-1)], dim=-1))

        if self.col_id_proj is not None:
            cells = cells + self.col_id_proj(
                self._column_ids(B, Fdim, cells.device, cells.dtype, column_id_seed)
            )[:, None, :, :]

        pad_rows = pad.repeat_interleave(N, dim=0)  # (B*N, F)
        # A row of all-padding would make softmax produce NaN over a fully masked sequence;
        # such a task is degenerate, but guard rather than emit NaN silently.
        safe_pad = pad_rows & ~pad_rows.all(dim=1, keepdim=True)

        if self.cfg.n_cell_blocks > 0:
            if self.cfg.cell_labels and y is not None:
                # §74/§76/task 39.2: labels reach column-level computation before pooling,
                # not only after it. Context cells get their true label broadcast across
                # every feature of that row; query cells get a learned mask token, exactly
                # mirroring query_token's role at the row level below. A quantile head's
                # label is the continuous target itself (task 48.3), not a class one-hot.
                if self.cfg.head_type == "quantile":
                    y_label = y[:, :n_ctx].to(cells.dtype).unsqueeze(-1)
                else:
                    y_label = F.one_hot(
                        y[:, :n_ctx].clamp(0, self.cfg.max_classes - 1), self.cfg.max_classes
                    ).to(cells.dtype)
                cell_y = torch.cat(
                    [
                        self.cell_y_proj(y_label),
                        self.cell_mask_token.expand(B, N - n_ctx, -1),
                    ],
                    dim=1,
                )  # (B, N, d_cell)
                cells = cells + cell_y[:, :, None, :]
            # attn_impl="split" needs no mask (AU5.2); "masked" builds the one it always has
            row_mask = (
                None if self.cfg.attn_impl == "split" else self._row_mask(N, n_ctx, X.device)
            )
            for _ in range(self.cfg.n_cell_blocks):
                # Row attention within one feature: a cell attends to the same feature's
                # cells in every other row -- a data-derived column identity, computed from
                # the column's own distribution rather than a random tag. (B,F,N,d) so the
                # mask, which does not depend on which feature, applies identically to all
                # B*F independent attention problems.
                by_feature = cells.permute(0, 2, 1, 3).reshape(B * Fdim, N, self.cfg.d_cell)
                by_feature = self._row_within_feature(by_feature, row_mask, B, Fdim, n_ctx)
                cells = by_feature.reshape(B, Fdim, N, self.cfg.d_cell).permute(0, 2, 1, 3)
                # Column attention within one row: the existing mechanism, reused per block.
                flat = cells.reshape(B * N, Fdim, self.cfg.d_cell)
                flat = self.column_encoder(flat, src_key_padding_mask=safe_pad)
                cells = flat.reshape(B, N, Fdim, self.cfg.d_cell)
        else:
            flat = cells.reshape(B * N, Fdim, self.cfg.d_cell)
            flat = self.column_encoder(flat, src_key_padding_mask=safe_pad)
            cells = flat.reshape(B, N, Fdim, self.cfg.d_cell)

        if self.cfg.n_summary_tokens > 1:
            # AU5.3: K summary tokens per row, concatenated, in place of mean+max.
            tokens = cells.reshape(B * N, Fdim, self.cfg.d_cell)
            q = self.summary_queries.expand(tokens.shape[0], -1, -1)
            summary, _ = self.summary_attn(
                q, tokens, tokens, key_padding_mask=safe_pad, need_weights=False
            )
            return self.row_proj(
                summary.reshape(B, N, self.cfg.n_summary_tokens * self.cfg.d_cell)
            )

        keep = (~pad)[:, None, :, None].to(cells.dtype)  # (B, 1, F, 1)
        denom = keep.sum(dim=2).clamp(min=1.0)
        pooled_mean = (cells * keep).sum(dim=2) / denom
        pooled_max = cells.masked_fill(keep == 0, float("-inf")).max(dim=2).values
        pooled_max = torch.nan_to_num(pooled_max, neginf=0.0)
        if self.cfg.pooling != "attention":
            return self.row_proj(torch.cat([pooled_mean, pooled_max], dim=-1))

        # attention pooling: one learned query over the feature tokens of every row
        q = self.pool_query.expand(flat.shape[0], -1, -1)
        attended, _ = self.pool_attn(
            q, flat, flat, key_padding_mask=safe_pad, need_weights=False
        )
        pooled_attn = attended.reshape(B, N, self.cfg.d_cell)
        return self.row_proj(
            torch.cat([pooled_mean, pooled_max, pooled_attn], dim=-1)
        )

    def _label_embedding(
        self,
        y: torch.Tensor,
        n_ctx: int,
        dtype: torch.dtype,
        period: torch.Tensor | None = None,
        hazard: bool = False,
    ) -> torch.Tensor:
        """Row-level label embedding: context rows get their label, query rows a mask token.

        The one place every row-stage path (:meth:`forward`, :meth:`term_structure`,
        :meth:`survival_loss`) builds it, so a change to what the context carries cannot
        reach one path and miss the others (``CLAUDE.md``, "a correction that lives in one
        method").

        Args:
            y: ``(B, N)`` labels; only the first ``n_ctx`` are read.
            n_ctx: Context/query split.
            dtype: Dtype of the row representations this is added to.
            period: ``(B, N)`` zero-based default period, ``CENSORED`` for none. Read only
                when ``cfg.hazard_period_context``, and then only its first ``n_ctx``
                entries, so a query row's period (its training target) never reaches any
                input.
            hazard: True from the hazard paths. The period embedding is a hazard-path
                input only; :meth:`forward` is the classification path and is unchanged by
                ``hazard_period_context`` (the training loop's held-out classification
                check calls it without periods).

        Returns:
            ``(B, N, d_model)``.

        Raises:
            ValueError: If ``hazard`` and ``cfg.hazard_period_context`` and ``period`` is
                ``None``. Falling
                back to the binary label alone would serve a model a context unlike the one
                it was trained on, with no error.
        """
        B, N = y.shape[0], y.shape[1]
        if self.cfg.head_type == "quantile":
            y_label = y[:, :n_ctx].to(dtype).unsqueeze(-1)
        else:
            y_label = F.one_hot(
                y[:, :n_ctx].clamp(0, self.cfg.max_classes - 1), self.cfg.max_classes
            ).to(dtype)
        ctx_emb = self.y_proj(y_label)
        use_period = hazard and self.cfg.hazard_period_context
        if use_period and period is not None:
            assert self.cfg.n_horizons is not None
            idx = period[:, :n_ctx].long().clamp(CENSORED, self.cfg.n_horizons - 1) - CENSORED
            ctx_emb = ctx_emb + self.period_embed(idx).to(dtype)
        elif use_period:
            raise ValueError(
                "this model was built with hazard_period_context=True, so its context rows "
                "need their default period: pass period=(B, N) with CENSORED (-1) for none"
            )
        return torch.cat([ctx_emb, self.query_token.expand(B, N - n_ctx, -1)], dim=1)

    def _row_attention(
        self, h: torch.Tensor, n_ctx: int, n_layers: int | None = None
    ) -> torch.Tensor:
        """Run the row (in-context) encoder, length-scaled when ``cfg.attn_length_scale``.

        The one place the row stage runs, for every path through the model and for the
        classifier's representation probe, so ``attn_length_scale`` and ``attn_impl`` reach
        all of them.

        Args:
            h: ``(B, N, d_model)`` row representations with label embeddings added.
            n_ctx: Context/query split.
            n_layers: Run only the first ``n_layers`` layers (an intermediate representation);
                ``None`` runs the whole stack.

        Returns:
            ``(B, N, d_model)``.
        """
        layers = self.encoder.layers if n_layers is None else self.encoder.layers[:n_layers]
        factor = (
            length_scale(n_ctx, self.cfg.attn_length_ref) if self.cfg.attn_length_scale else 1.0
        )
        if self.cfg.attn_impl == "split":
            return self._prenorm_stack(layers, h, n_ctx, None, factor)
        mask = self._row_mask(h.shape[1], n_ctx, h.device)
        if not self.cfg.attn_length_scale:
            if n_layers is None:
                return self.encoder(h, mask=mask)
            for layer in layers:
                h = layer(h, src_mask=mask)
            return h
        return self._prenorm_stack(layers, h, n_ctx, ~mask, factor)

    def _prenorm_stack(
        self,
        layers: nn.ModuleList,
        h: torch.Tensor,
        n_ctx: int,
        allowed: torch.Tensor | None,
        factor: float,
    ) -> torch.Tensor:
        """Run encoder layers with the attention computed here rather than by the module.

        The same pre-norm layer as ``nn.TransformerEncoderLayer(norm_first=True)``, written
        out so the attention logits can carry the length factor (AU5.17) or the attention can
        run unmasked (AU5.2). It reads the stock layers' own parameters, so neither flag adds
        a weight and a checkpoint trained either way loads either way.

        Args:
            layers: The ``nn.TransformerEncoderLayer`` modules to run, in order.
            h: ``(B, N, D)`` inputs.
            n_ctx: Context/query split.
            allowed: ``(N, N)`` boolean mask, True = may attend, for the masked path; ``None``
                for :func:`split_row_attention`.
            factor: Multiplier on ``q.k / sqrt(d_head)``.

        Returns:
            ``(B, N, D)``.
        """
        for layer in layers:
            h = h + layer.dropout1(
                self._scaled_self_attention(
                    layer.self_attn, layer.norm1(h), allowed, factor, n_ctx=n_ctx
                )
            )
            ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm2(h)))))
            h = h + layer.dropout2(ff)
        return h

    def _scaled_self_attention(
        self,
        mha: nn.MultiheadAttention,
        x: torch.Tensor,
        allowed: torch.Tensor | None,
        factor: float,
        n_ctx: int | None = None,
    ) -> torch.Tensor:
        """Self-attention through ``mha``'s weights with logits multiplied by ``factor``.

        Args:
            mha: The stock attention module whose projections are used.
            x: ``(B, N, D)`` normalised inputs.
            allowed: ``(N, N)`` boolean mask, True = may attend (the inverse of
                :meth:`_row_mask`'s convention, as ``scaled_dot_product_attention`` expects).
                ``None`` computes the same context-plus-self attention through
                :func:`split_row_attention`, which needs ``n_ctx``.
            factor: Multiplier on ``q.k / sqrt(d_head)``; 1 reproduces ``mha``.
            n_ctx: Context/query split; read only when ``allowed`` is ``None``.

        Returns:
            ``(B, N, D)``.
        """
        B, N, D = x.shape
        H = mha.num_heads
        dh = D // H
        q, k, v = F.linear(x, mha.in_proj_weight, mha.in_proj_bias).chunk(3, dim=-1)
        q, k, v = (t.reshape(B, N, H, dh).transpose(1, 2) for t in (q, k, v))
        dropout_p = mha.dropout if self.training else 0.0
        if allowed is None:
            assert n_ctx is not None
            out = split_row_attention(q, k, v, n_ctx, factor / math.sqrt(dh), dropout_p)
        else:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=allowed,
                dropout_p=dropout_p,
                scale=factor / math.sqrt(dh),
            )
        return mha.out_proj(out.transpose(1, 2).reshape(B, N, D))

    def forward(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        n_ctx: int,
        n_classes: torch.Tensor | None = None,
        column_id_seed: int | None = None,
    ) -> torch.Tensor:
        """Compute class logits, or quantile predictions, for every row.

        Args:
            X: ``(B, N, F)`` raw features, NaN for missing cells and padded columns.
            y: ``(B, N)`` labels; only the first ``n_ctx`` are read. Integer class indices
                for ``head_type="binned"``, a continuous (context-normalised) target for
                ``head_type="quantile"``.
            n_ctx: Context/query split.
            n_classes: Optional ``(B,)`` valid class counts; logits for classes a task does
                not have become ``-inf``. Ignored for ``head_type="quantile"``, which has no
                class-count concept.

        Returns:
            ``(B, N, max_classes)`` logits for ``"binned"``, or ``(B, N, n_quantiles)``
            continuous quantile predictions for ``"quantile"``. Only rows at or past
            ``n_ctx`` are predictions either way.
        """
        h = self.encode_rows(X, n_ctx, column_id_seed=column_id_seed, y=y)
        h = h + self._label_embedding(y, n_ctx, h.dtype)
        h = self._row_attention(h, n_ctx)
        out = self.head(self.norm(h))
        if self.cfg.head_type == "quantile":
            return out
        if n_classes is not None:
            valid = (
                torch.arange(self.cfg.max_classes, device=X.device)[None, None, :]
                < n_classes[:, None, None]
            )
            out = out.masked_fill(~valid, float("-inf"))
        return out

    @torch.no_grad()
    def predict_context(
        self,
        Xc: np.ndarray,
        yc: np.ndarray,
        Xq: np.ndarray,
        *,
        n_classes: int | None = None,
        column_id_seed: int | None = None,
        output: ContextOutput = "logits",
        stage: RepresentationStage | None = None,
        period: np.ndarray | None = None,
        pad_features: bool = True,
    ) -> torch.Tensor:
        """Score query rows against one context: the single inference entry point (DR1.4).

        Every inference caller -- the classifier's ``predict_proba``, ``predict_term_structure``
        and ``transform_representation``, the quantile regressor and the capability probes --
        comes through here, so the padding, the label dtype, the device and the column
        identities are decided once. Before this, five copies did it, and the term-structure
        copy never passed ``column_id_seed``: at ``random_state != 0`` its column identities
        came from seed 0 while ``predict_proba``'s came from ``random_state``.

        Inputs are taken as they are: no conditioning and no base-rate correction, which stay
        with the estimators that own them.

        Args:
            Xc: ``(n_ctx, f)`` context rows, as the caller conditioned them.
            yc: ``(n_ctx,)`` context labels: class indices, or the normalised continuous
                target for ``head_type="quantile"``.
            Xq: ``(n_q, f)`` query rows, conditioned as ``Xc`` was.
            n_classes: Valid classes for ``output="logits"``; logits past it are ``-inf``.
                ``None`` masks nothing (and a quantile head ignores it).
            column_id_seed: Seed for the column identities; see :meth:`encode_rows`.
            output: ``"logits"`` (:meth:`forward`'s output), ``"term_structure"``
                (:meth:`term_structure`'s cumulative PD) or ``"representation"`` (a row
                representation at ``stage``).
            stage: For ``output="representation"`` only, and required there:
                ``"encode_rows"`` (:meth:`encode_rows` alone), ``"encoder_mid"`` (plus the
                label embedding and the first half of the row encoder's layers, rounded up,
                with no final norm) or ``"pre_head"`` (everything :meth:`forward` runs short
                of ``self.head``).
            period: ``(n_ctx,)`` context default periods for ``output="term_structure"``;
                required by a ``hazard_period_context`` checkpoint, ignored otherwise.
            pad_features: NaN-pad the feature axis to ``cfg.max_features`` (True, what every
                estimator does) or keep the width ``f`` the caller gave. The capability
                probes pass False: they have always fed the model their raw, unpadded probe
                features, and the column-identity draw depends on the width, so padding would
                move every number they have recorded.

        Returns:
            Query rows only, on the model's device: ``(n_q, max_classes)`` logits or
            ``(n_q, n_quantiles)`` quantiles, ``(n_q, n_horizons)`` cumulative PD, or
            ``(n_q, d_model)`` representations.

        Raises:
            ValueError: If the rows are wider than ``cfg.max_features``, ``stage`` is missing
                for or given without ``output="representation"``, or ``output`` is unknown.
            RuntimeError: From :meth:`term_structure` without a hazard head.
        """
        if output not in ("logits", "term_structure", "representation"):
            raise ValueError(f"unknown output {output!r}")
        if (stage is None) != (output != "representation"):
            raise ValueError("stage is required with output='representation' and only there")
        if Xc.shape[1] > self.cfg.max_features or Xq.shape[1] > self.cfg.max_features:
            raise ValueError(
                f"model supports at most {self.cfg.max_features} features, got "
                f"{max(Xc.shape[1], Xq.shape[1])}"
            )
        width = self.cfg.max_features if pad_features else Xc.shape[1]
        y_dtype = np.float32 if self.cfg.head_type == "quantile" else np.int64
        Xp, yp, pp = pack_context(
            Xc, yc, Xq, width, y_dtype, period if output == "term_structure" else None
        )
        # derived from the model, not passed alongside it (CLAUDE.md, "two ways a long run
        # lies about itself")
        device = next(self.parameters()).device
        X = torch.from_numpy(Xp).to(device)
        y = torch.from_numpy(yp).to(device)
        n_ctx = Xc.shape[0]
        if output == "term_structure":
            p = None if pp is None else torch.from_numpy(pp).to(device)
            return self.term_structure(X, y, n_ctx, period=p, column_id_seed=column_id_seed)[0]
        if output == "logits":
            nc = None if n_classes is None else torch.tensor([n_classes], device=device)
            return self(X, y, n_ctx, nc, column_id_seed=column_id_seed)[0, n_ctx:]
        h = self.encode_rows(X, n_ctx, column_id_seed=column_id_seed, y=y)
        if stage == "encoder_mid":
            # Half the row-to-row stack, rounded up -- partway through, not before or after
            # all of it. No final `self.norm`: that normalisation is shaped for `self.head`'s
            # input, not for a linear probe reading an intermediate layer.
            n_half = (len(self.encoder.layers) + 1) // 2
            h = self._row_attention(h + self._label_embedding(y, n_ctx, h.dtype), n_ctx, n_half)
        elif stage == "pre_head":
            h = self.norm(self._row_attention(h + self._label_embedding(y, n_ctx, h.dtype), n_ctx))
        elif stage != "encode_rows":
            raise ValueError(f"unknown stage {stage!r}")
        return h[0, n_ctx:]

    def loss(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        n_ctx: int,
        n_classes: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Mean cross-entropy, or mean pinball loss, over query rows.

        Cross-entropy is a *proper scoring rule*, so minimising it rewards calibrated
        probabilities rather than merely correct rankings. That matters here: a credit model
        is judged on whether a stated 2% probability of default happens about 2% of the time,
        which AUC cannot see at all. See ``metrics.py``.

        The pinball (quantile) loss is the continuous analogue: each output column is
        trained against its own quantile level ``tau`` so that minimising it is, for that
        column alone, minimised exactly at the ``tau``-quantile of the target's conditional
        distribution -- the single-column generalisation of what cross-entropy does for a
        full categorical distribution.

        ``weights`` (``(B, n_queries)``, cross-entropy only) gives each query row a weight and
        returns the weighted mean. Used by fine-tuning's importance-weighted, default-enriched
        queries (§164): oversampling the rare class and weighting it back by true/sampled
        rate keeps the loss an unbiased estimate of the uniform-query loss. ``None`` is the
        plain mean, unchanged.
        """
        out = self.forward(X, y, n_ctx, n_classes)[:, n_ctx:]
        if self.cfg.head_type == "quantile":
            if weights is not None:
                raise ValueError("per-row weights are implemented for cross-entropy only")
            target = y[:, n_ctx:].to(out.dtype).unsqueeze(-1)  # (B, Nq, 1)
            diff = target - out  # (B, Nq, n_quantiles)
            taus = self.quantile_taus.to(out.dtype)
            return torch.maximum(taus * diff, (taus - 1.0) * diff).mean()
        if weights is None:
            return F.cross_entropy(
                out.reshape(-1, self.cfg.max_classes), y[:, n_ctx:].reshape(-1)
            )
        per_row = F.cross_entropy(
            out.reshape(-1, self.cfg.max_classes), y[:, n_ctx:].reshape(-1), reduction="none"
        )
        w = weights.reshape(-1).to(per_row.dtype)
        return (per_row * w).sum() / w.sum()

    def num_parameters(self) -> int:
        """Total trainable parameters.

        Reported in every training log because a configuration can silently come out the wrong
        size: ``--d-ff`` did not exist once and the feed-forward width stayed pinned at 512
        while ``d_model`` grew, producing a 3.3M "medium" instead of 4.9M
        (``docs/infra/HF_JOBS.md``). Nothing failed; the number was simply not looked at.

        Returns:
            Parameter count.
        """
        return sum(p.numel() for p in self.parameters())

    def save(
        self,
        path: str,
        trained_objectives: tuple[str, ...] = (),
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        """Write the checkpoint, recording which objectives were actually optimised.

        Args:
            path: Destination file.
            trained_objectives: Names of the objectives this checkpoint was trained on, from
                ``{"classification", "quantile", "survival"}``. **Recording this is not
                bookkeeping.**
                The training loop optimises the survival loss *or* the classification loss,
                never both, so a hazard checkpoint's classification head is still at random
                initialisation — and ``predict_proba`` on it returned confident nonsense with
                no error at all (mean predicted 0.69 against a 4.7% base rate, AUC 0.37).
                An empty tuple means the checkpoint predates this field and nothing can be
                assumed about it.
            provenance: What produced the weights (design review DR1.2): the training loop
                passes the prior and training configurations, the realised source shares,
                the tasks seen and the task dump used. Written under the checkpoint's
                ``provenance`` key together with the model configuration, the commit, the
                dirty flag and the ``fintfm`` and ``torch`` versions, which are recorded on
                every save, so even a checkpoint saved outside the training loop says which
                code wrote it. :meth:`load` never reads the key, so it changes no output;
                :func:`fintfm.provenance.read_checkpoint_provenance` reads it back.
        """
        from fintfm.provenance import environment_record, json_safe

        record = {
            **environment_record(),
            "model_config": json_safe(asdict(self.cfg)),
            **json_safe(dict(provenance or {})),
        }
        torch.save(
            {
                "config": asdict(self.cfg),
                "state_dict": self.state_dict(),
                "trained_objectives": list(trained_objectives),
                "provenance": record,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, map_location: str | torch.device = "cpu") -> FinancialTFM:
        """Rebuild a model from a checkpoint, in eval mode.

        Args:
            path: Checkpoint written by :meth:`save`.
            map_location: Torch device to load onto.

        Returns:
            The model, in ``eval()`` mode, carrying ``trained_objectives`` so
            :meth:`assert_trained_for` can refuse a head that was never optimised. The
            checkpoint's ``provenance`` record, if any, is deliberately not read: it
            describes the weights and must never change what they compute.
        """
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        cfg = dict(ckpt["config"])
        # Checkpoints written before §54 have no column identities and no ``col_id_proj``
        # weights. Defaulting them to the new behaviour would fail to load; defaulting them
        # to 0 keeps them usable as the "before" arm of the comparison that motivated it.
        if "column_id_dim" not in cfg:
            cfg["column_id_dim"] = 0
        model = cls(ModelConfig(**cfg))
        model.load_state_dict(ckpt["state_dict"])
        model.trained_objectives = tuple(ckpt.get("trained_objectives", ()))
        return model.eval()

    def assert_trained_for(self, objective: str) -> None:
        """Refuse to serve predictions from a head that was never trained.

        Args:
            objective: ``"classification"``, ``"quantile"`` or ``"survival"``.

        Raises:
            RuntimeError: If the checkpoint records objectives and this is not among them.
                A checkpoint with no record (an older file, or a model built in memory) is
                allowed through, because refusing would break every in-process use.
        """
        known = getattr(self, "trained_objectives", ())
        if known and objective not in known:
            raise RuntimeError(
                f"this checkpoint was trained on {sorted(known)} and not on {objective!r}, "
                f"so the {objective} head is at its random initialisation; predictions from "
                "it are meaningless. Pretrain without --n-horizons for classification, or "
                "use predict_term_structure for a hazard checkpoint."
            )

    def term_structure(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        n_ctx: int,
        period: torch.Tensor | None = None,
        column_id_seed: int | None = None,
    ) -> torch.Tensor:
        """Cumulative PD across horizons for the query rows, monotone by construction.

        Args:
            X: ``(B, N, F)`` raw features, NaN for missing and padded.
            y: ``(B, N)`` labels; only the first ``n_ctx`` are read.
            n_ctx: Context/query split.
            period: ``(B, N)`` default periods, ``CENSORED`` for none; only the first
                ``n_ctx`` are read. Required when ``cfg.hazard_period_context``, ignored
                otherwise.
            column_id_seed: Seed for the column identities, as in :meth:`encode_rows`.
                ``None`` keeps the fixed eval-mode draw (seed 0) every earlier caller got.

        Returns:
            ``(B, N - n_ctx, n_horizons)`` cumulative default probability, non-decreasing in
            the horizon for every row. See ``fintfm.modeling.hazard``.

        Raises:
            RuntimeError: If the model was built without ``n_horizons``.
            ValueError: If ``cfg.hazard_period_context`` and no ``period`` is given.
        """
        if self.hazard is None:
            raise RuntimeError(
                "this model has no hazard head; build it with ModelConfig(n_horizons=K)"
            )
        rows = self.encode_rows(X, n_ctx, column_id_seed=column_id_seed, y=y)
        y_emb = self._label_embedding(y, n_ctx, rows.dtype, period=period, hazard=True)
        h = self._row_attention(rows + y_emb, n_ctx)
        return self.hazard.cumulative_pd(self.norm(h)[:, n_ctx:])

    def survival_loss(
        self, X: torch.Tensor, y: torch.Tensor, period: torch.Tensor, n_ctx: int
    ) -> torch.Tensor:
        """Discrete-time survival negative log-likelihood over the query rows.

        Fits the whole term structure at once rather than one horizon at a time, which is
        what makes the horizons mutually consistent (``docs/results/FINDINGS.md`` §20). Independent
        per-horizon models cannot do this in principle, since they share no parameters.

        Args:
            X: ``(B, N, F)`` raw features.
            y: ``(B, N)`` binary labels, used only to embed the context.
            period: ``(B, N)`` zero-based default period, ``-1`` for censored. The query
                rows' entries are the targets; with ``cfg.hazard_period_context`` the context
                rows' entries are also embedded as inputs, and only those.
            n_ctx: Context/query split.

        Returns:
            Scalar mean negative log-likelihood over query rows.

        Raises:
            RuntimeError: If the model has no hazard head.
        """
        if self.hazard is None:
            raise RuntimeError(
                "this model has no hazard head; build it with ModelConfig(n_horizons=K)"
            )
        rows = self.encode_rows(X, n_ctx, y=y)
        y_emb = self._label_embedding(y, n_ctx, rows.dtype, period=period, hazard=True)
        h = self._row_attention(rows + y_emb, n_ctx)
        h = self.norm(h)[:, n_ctx:]
        return self.hazard.loss(h.reshape(-1, self.cfg.d_model), period[:, n_ctx:].reshape(-1))
