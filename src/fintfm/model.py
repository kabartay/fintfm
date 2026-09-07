"""Prior-fitted network for tabular in-context classification.

Rows are tokens. Context rows carry a feature embedding plus a label
embedding; query rows carry only the feature embedding. A Transformer encoder
with a row-level attention mask lets every row attend to context rows only,
so predictions for one query never depend on other queries. A linear head
produces logits over ``max_classes``; classes beyond a task's ``n_classes``
are masked out. This is an independent implementation of the PFN idea, not
derived from any existing codebase.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """Architecture hyper-parameters.

    Attributes:
        max_features: Padded feature width the encoder accepts.
        max_classes: Width of the classification head.
        d_model: Transformer width.
        n_heads: Attention heads.
        n_layers: Encoder layers.
        d_ff: Feed-forward width.
        dropout: Dropout probability (0 for pretraining on infinite synthetic data).
    """

    max_features: int = 24
    max_classes: int = 10
    d_model: int = 192
    n_heads: int = 4
    n_layers: int = 6
    d_ff: int = 512
    dropout: float = 0.0


def normalize_features(X: torch.Tensor, n_ctx: int) -> torch.Tensor:
    """Z-score features per task using *context* statistics; clip; NaN -> 0 plus mask channel.

    Args:
        X: ``(B, N, F)`` with NaN for missing/padded cells.
        n_ctx: Number of leading context rows whose statistics are used.

    Returns:
        ``(B, N, 2F)`` tensor: normalised values (missing set to 0) followed by
        a 0/1 missingness indicator per feature.
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
    return torch.cat([Z, missing.to(Z.dtype)], dim=-1)


class FinancialTFM(nn.Module):
    """Tabular in-context classifier.

    Args:
        cfg: :class:`ModelConfig`.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.x_proj = nn.Sequential(
            nn.Linear(2 * cfg.max_features, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.y_proj = nn.Linear(cfg.max_classes, cfg.d_model, bias=False)
        self.query_token = nn.Parameter(torch.zeros(cfg.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.max_classes)

    @staticmethod
    def _attn_mask(n_rows: int, n_ctx: int, device: torch.device) -> torch.Tensor:
        """Boolean ``(N, N)`` mask, True = blocked. Everyone attends to context only."""
        mask = torch.zeros(n_rows, n_rows, dtype=torch.bool, device=device)
        mask[:, n_ctx:] = True
        return mask

    def forward(
        self, X: torch.Tensor, y: torch.Tensor, n_ctx: int, n_classes: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute class logits for every row.

        Args:
            X: ``(B, N, F)`` raw features, NaN for missing/padded.
            y: ``(B, N)`` labels; only the first ``n_ctx`` are used.
            n_ctx: Context/query split.
            n_classes: Optional ``(B,)`` valid class counts; invalid logits become ``-inf``.

        Returns:
            ``(B, N, max_classes)`` logits. Only rows ``>= n_ctx`` are meaningful predictions.
        """
        B, N, _ = X.shape
        h = self.x_proj(normalize_features(X, n_ctx))
        y_onehot = F.one_hot(y[:, :n_ctx].clamp(0, self.cfg.max_classes - 1), self.cfg.max_classes).to(h.dtype)
        y_emb = torch.cat(
            [self.y_proj(y_onehot), self.query_token.expand(B, N - n_ctx, -1)], dim=1
        )
        h = h + y_emb
        h = self.encoder(h, mask=self._attn_mask(N, n_ctx, X.device))
        logits = self.head(self.norm(h))
        if n_classes is not None:
            valid = torch.arange(self.cfg.max_classes, device=X.device)[None, None, :] < n_classes[:, None, None]
            logits = logits.masked_fill(~valid, float("-inf"))
        return logits

    def loss(self, X: torch.Tensor, y: torch.Tensor, n_ctx: int, n_classes: torch.Tensor) -> torch.Tensor:
        """Mean cross-entropy over query rows."""
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
