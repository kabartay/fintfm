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
described in the public TabPFN / TabICL / TabFM literature (see ``docs/REFERENCES.md``). No
code or weights from those projects are used; see the licensing boundary in ``CLAUDE.md``.

Checkpoints written by the earlier flat-vector architecture are not loadable here. They were
smoke tests only and none were kept.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from fintfm.modeling.hazard import HazardHead

# Seed used for column identities whenever a caller does not supply one and the model is in
# eval mode. Its value is arbitrary; that it is *fixed* is the point (see _column_ids).
_DEFAULT_COLUMN_ID_SEED = 0


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
            (``docs/FINDINGS.md`` §50). A mean is weight-blind: every feature token
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
            of the multiset of that row's values* (``docs/FINDINGS.md`` §54, verified: the
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
            incoherence measured in ``docs/FINDINGS.md`` §11. ``None`` keeps the model
            classification-only.
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
              across the whole task and are therefore padding rather than data.
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
    pad = torch.isnan(X).all(dim=1)
    return Z, missing.to(Z.dtype), pad


class FinancialTFM(nn.Module):
    """Tabular in-context classifier.

    Args:
        cfg: :class:`ModelConfig`.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.cell_embed = nn.Sequential(
            nn.Linear(2, cfg.d_cell),
            nn.GELU(),
            nn.Linear(cfg.d_cell, cfg.d_cell),
        )
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
            nhead=cfg.n_heads,
            dim_feedforward=2 * cfg.d_cell,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.column_encoder = nn.TransformerEncoder(
            col_layer, num_layers=cfg.n_col_layers, enable_nested_tensor=False
        )
        # Pooling over columns is a masked mean (order-invariant) plus a masked max, which
        # keeps a signal a mean washes out: one extreme ratio in an otherwise ordinary firm.
        if cfg.pooling not in ("meanmax", "attention"):
            raise ValueError(
                f"pooling must be 'meanmax' or 'attention', got {cfg.pooling!r}"
            )
        if cfg.pooling == "attention":
            # A single learned query attends over the feature tokens, so the reduction can
            # weight features instead of averaging them (§50). Mean and max are kept
            # alongside: they are cheap, they carry the summary statistics attention would
            # otherwise have to rediscover, and keeping them makes the change additive rather
            # than a replacement whose regressions would be hard to attribute.
            self.pool_query = nn.Parameter(torch.randn(1, 1, cfg.d_cell) * 0.02)
            self.pool_attn = nn.MultiheadAttention(
                cfg.d_cell, cfg.n_heads, dropout=cfg.dropout, batch_first=True
            )
            self.row_proj = nn.Linear(3 * cfg.d_cell, cfg.d_model)
        else:
            self.row_proj = nn.Linear(2 * cfg.d_cell, cfg.d_model)
        self.y_proj = nn.Linear(cfg.max_classes, cfg.d_model, bias=False)
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
        self.head = nn.Linear(cfg.d_model, cfg.max_classes)
        self.hazard = (
            HazardHead(cfg.d_model, cfg.n_horizons) if cfg.n_horizons is not None else None
        )

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

    def _column_ids(
        self, B: int, Fdim: int, device: torch.device, dtype: torch.dtype, seed: int | None
    ) -> torch.Tensor:
        """Draw the random per-task column identity vectors.

        **Random while training, deterministic while evaluating.** Training wants a fresh
        draw every step, because it is the resampling that stops any column index acquiring a
        fixed meaning and so keeps the task distribution column-order invariant. Inference
        wants reproducibility: a credit model whose score changes between two identical calls
        fails model validation before anyone looks at its accuracy, and stochastic predictions
        are exactly what ``docs/FINDINGS.md`` §55 argues supervisors grade against.

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
        self, X: torch.Tensor, n_ctx: int, column_id_seed: int | None = None
    ) -> torch.Tensor:
        """Turn raw cells into one vector per row.

        Args:
            X: ``(B, N, F)`` raw features.
            n_ctx: Context/query split, used for normalisation statistics.
            column_id_seed: Seed for the random column identities. ``None`` means a fresh
                draw while training and a fixed one while evaluating; pass distinct integers
                to ensemble over draws.

        Returns:
            ``(B, N, d_model)`` row representations.
        """
        B, N, Fdim = X.shape
        Z, missing, pad = normalize_features(X, n_ctx)
        cells = self.cell_embed(torch.stack([Z, missing], dim=-1))  # (B, N, F, d_cell)

        if self.col_id_proj is not None:
            cells = cells + self.col_id_proj(
                self._column_ids(B, Fdim, cells.device, cells.dtype, column_id_seed)
            )[:, None, :, :]

        flat = cells.reshape(B * N, Fdim, self.cfg.d_cell)
        pad_rows = pad.repeat_interleave(N, dim=0)  # (B*N, F)
        # A row of all-padding would make softmax produce NaN over a fully masked sequence;
        # such a task is degenerate, but guard rather than emit NaN silently.
        safe_pad = pad_rows & ~pad_rows.all(dim=1, keepdim=True)
        flat = self.column_encoder(flat, src_key_padding_mask=safe_pad)
        cells = flat.reshape(B, N, Fdim, self.cfg.d_cell)

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

    def forward(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        n_ctx: int,
        n_classes: torch.Tensor | None = None,
        column_id_seed: int | None = None,
    ) -> torch.Tensor:
        """Compute class logits for every row.

        Args:
            X: ``(B, N, F)`` raw features, NaN for missing cells and padded columns.
            y: ``(B, N)`` labels; only the first ``n_ctx`` are read.
            n_ctx: Context/query split.
            n_classes: Optional ``(B,)`` valid class counts; logits for classes a task does
                not have become ``-inf``.

        Returns:
            ``(B, N, max_classes)`` logits. Only rows at or past ``n_ctx`` are predictions.
        """
        B, N, _ = X.shape
        h = self.encode_rows(X, n_ctx, column_id_seed=column_id_seed)
        y_onehot = F.one_hot(
            y[:, :n_ctx].clamp(0, self.cfg.max_classes - 1), self.cfg.max_classes
        ).to(h.dtype)
        y_emb = torch.cat(
            [self.y_proj(y_onehot), self.query_token.expand(B, N - n_ctx, -1)], dim=1
        )
        h = h + y_emb
        h = self.encoder(h, mask=self._row_mask(N, n_ctx, X.device))
        logits = self.head(self.norm(h))
        if n_classes is not None:
            valid = (
                torch.arange(self.cfg.max_classes, device=X.device)[None, None, :]
                < n_classes[:, None, None]
            )
            logits = logits.masked_fill(~valid, float("-inf"))
        return logits

    def loss(
        self, X: torch.Tensor, y: torch.Tensor, n_ctx: int, n_classes: torch.Tensor
    ) -> torch.Tensor:
        """Mean cross-entropy over query rows.

        Cross-entropy is a *proper scoring rule*, so minimising it rewards calibrated
        probabilities rather than merely correct rankings. That matters here: a credit model
        is judged on whether a stated 2% probability of default happens about 2% of the time,
        which AUC cannot see at all. See ``metrics.py``.
        """
        logits = self.forward(X, y, n_ctx, n_classes)[:, n_ctx:]
        return F.cross_entropy(logits.reshape(-1, self.cfg.max_classes), y[:, n_ctx:].reshape(-1))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save(self, path: str, trained_objectives: tuple[str, ...] = ()) -> None:
        """Write the checkpoint, recording which objectives were actually optimised.

        Args:
            path: Destination file.
            trained_objectives: Names of the objectives this checkpoint was trained on, from
                ``{"classification", "survival"}``. **Recording this is not bookkeeping.**
                The training loop optimises the survival loss *or* the classification loss,
                never both, so a hazard checkpoint's classification head is still at random
                initialisation — and ``predict_proba`` on it returned confident nonsense with
                no error at all (mean predicted 0.69 against a 4.7% base rate, AUC 0.37).
                An empty tuple means the checkpoint predates this field and nothing can be
                assumed about it.
        """
        torch.save(
            {
                "config": asdict(self.cfg),
                "state_dict": self.state_dict(),
                "trained_objectives": list(trained_objectives),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, map_location: str | torch.device = "cpu") -> FinancialTFM:
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
            objective: ``"classification"`` or ``"survival"``.

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
        self, X: torch.Tensor, y: torch.Tensor, n_ctx: int
    ) -> torch.Tensor:
        """Cumulative PD across horizons for the query rows, monotone by construction.

        Args:
            X: ``(B, N, F)`` raw features, NaN for missing and padded.
            y: ``(B, N)`` labels; only the first ``n_ctx`` are read.
            n_ctx: Context/query split.

        Returns:
            ``(B, N - n_ctx, n_horizons)`` cumulative default probability, non-decreasing in
            the horizon for every row. See ``fintfm.modeling.hazard``.

        Raises:
            RuntimeError: If the model was built without ``n_horizons``.
        """
        if self.hazard is None:
            raise RuntimeError(
                "this model has no hazard head; build it with ModelConfig(n_horizons=K)"
            )
        rows = self.encode_rows(X, n_ctx)
        y_onehot = F.one_hot(
            y[:, :n_ctx].clamp(0, self.cfg.max_classes - 1), self.cfg.max_classes
        ).to(rows.dtype)
        y_emb = torch.cat(
            [self.y_proj(y_onehot), self.query_token.expand(X.shape[0], X.shape[1] - n_ctx, -1)],
            dim=1,
        )
        h = self.encoder(rows + y_emb, mask=self._row_mask(X.shape[1], n_ctx, X.device))
        return self.hazard.cumulative_pd(self.norm(h)[:, n_ctx:])

    def survival_loss(
        self, X: torch.Tensor, y: torch.Tensor, period: torch.Tensor, n_ctx: int
    ) -> torch.Tensor:
        """Discrete-time survival negative log-likelihood over the query rows.

        Fits the whole term structure at once rather than one horizon at a time, which is
        what makes the horizons mutually consistent (``docs/FINDINGS.md`` §20). Independent
        per-horizon models cannot do this in principle, since they share no parameters.

        Args:
            X: ``(B, N, F)`` raw features.
            y: ``(B, N)`` binary labels, used only to embed the context.
            period: ``(B, N)`` zero-based default period, ``-1`` for censored.
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
        rows = self.encode_rows(X, n_ctx)
        y_onehot = F.one_hot(
            y[:, :n_ctx].clamp(0, self.cfg.max_classes - 1), self.cfg.max_classes
        ).to(rows.dtype)
        y_emb = torch.cat(
            [self.y_proj(y_onehot), self.query_token.expand(X.shape[0], X.shape[1] - n_ctx, -1)],
            dim=1,
        )
        h = self.encoder(rows + y_emb, mask=self._row_mask(X.shape[1], n_ctx, X.device))
        h = self.norm(h)[:, n_ctx:]
        return self.hazard.loss(h.reshape(-1, self.cfg.d_model), period[:, n_ctx:].reshape(-1))
