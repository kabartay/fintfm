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

    def encode_rows(self, X: torch.Tensor, n_ctx: int) -> torch.Tensor:
        """Turn raw cells into one vector per row, invariant to column order.

        Args:
            X: ``(B, N, F)`` raw features.
            n_ctx: Context/query split, used for normalisation statistics.

        Returns:
            ``(B, N, d_model)`` row representations.
        """
        B, N, Fdim = X.shape
        Z, missing, pad = normalize_features(X, n_ctx)
        cells = self.cell_embed(torch.stack([Z, missing], dim=-1))  # (B, N, F, d_cell)

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
        return self.row_proj(torch.cat([pooled_mean, pooled_max], dim=-1))

    def forward(
        self, X: torch.Tensor, y: torch.Tensor, n_ctx: int, n_classes: torch.Tensor | None = None
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
        h = self.encode_rows(X, n_ctx)
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

    def save(self, path: str) -> None:
        torch.save({"config": asdict(self.cfg), "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, map_location: str | torch.device = "cpu") -> FinancialTFM:
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(ModelConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["state_dict"])
        return model.eval()
