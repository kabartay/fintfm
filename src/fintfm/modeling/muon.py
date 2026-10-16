"""Muon, an opt-in pretraining optimiser (audit AU5.17), with an AdamW side for the rest.

Muon (Jordan et al., 2024, "Muon: An optimizer for hidden layers in neural networks",
kellerjordan.github.io/posts/muon; reference implementation MIT-licensed) takes SGD-momentum's
update for a 2-D weight matrix and replaces it with the nearest semi-orthogonal matrix,
computed by a few Newton-Schulz iterations. Written here from that description and from Liu
et al., 2025, "Muon is Scalable for LLM Training" (Moonlight, arXiv:2502.16982) for the
update scaling and weight decay; nothing is taken from any tabular-foundation-model project.

The write-up is explicit that Muon is for **hidden** weight matrices only: input embeddings,
the output head, and every gain or bias belong to a standard optimiser. So one
:class:`Muon` instance carries both kinds of parameter group, flagged by ``use_muon``, and
runs AdamW on the second kind. One optimiser object rather than two keeps the training loop's
single ``opt``/``sched``/``state_dict`` path unchanged: ``LambdaLR`` scales every group's rate
by the same warmup-and-cosine multiplier, and a resume restores both halves from one dict.

Design review DR1.7 (2026-10-16) set the defaults before any run used Muon:

- **Update scale** ``"rms_match"`` (default) multiplies each orthogonalised ``n x m`` update
  by ``0.2·sqrt(max(n, m))``. A semi-orthogonal matrix has per-element RMS
  ``1/sqrt(max(n, m))``, so the product's RMS is 0.2 whatever the shape: Moonlight's choice,
  matched to AdamW's typical steady-state update RMS so that Muon can share AdamW's rate.
  ``"keller"`` is the reference implementation's ``sqrt(max(1, n/m))``, meant for its lr of
  0.02; at that lr a matrix moves about ten times as far per element as under ``rms_match``
  at an AdamW rate (the review's arithmetic, ESTIMATED).
- **Weight decay** is decoupled, ``w <- w·(1 - lr·wd)``, on Muon's matrices as on AdamW's,
  which Moonlight found necessary once training runs long.
- **Fused projections are split.** ``row_blocks=3`` orthogonalises a fused QKV
  ``in_proj_weight`` as its Q, K and V row blocks, each scaled by its own shape: the
  write-up reports Muon working better on Q, K and V separately than on the fused matrix.
  Not per head: neither source describes per-head orthogonalisation, and Q, K and V are
  three linear maps stored together for speed, where a head is a slice of one of them.
- **Newton-Schulz precision** is bf16 only where bf16 is native (CUDA compute capability
  8.0+), float32 elsewhere (:func:`newton_schulz_dtype`).

Not the default, and no accuracy effect is claimed for it: ``--optimizer adamw`` remains what
every checkpoint trained with.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch

#: Coefficients of the quintic Newton-Schulz iteration from the write-up. Chosen there to
#: push every singular value into roughly [0.7, 1.2] in five steps rather than to converge
#: exactly to 1, which the write-up measured as not hurting the optimiser.
_NS_COEFFS = (3.4445, -4.7750, 2.0315)

#: Accepted update-scaling rules; see the module docstring.
MUON_SCALES = ("rms_match", "keller")

#: The per-element update RMS ``"rms_match"`` targets (Moonlight's matched-to-AdamW value).
_RMS_MATCH_TARGET = 0.2


def newton_schulz_dtype(device: torch.device) -> torch.dtype:
    """The precision Newton-Schulz runs in for matrices on ``device``.

    bf16 on CUDA devices of compute capability 8.0+ (A10G, L4, A100), where it is native and
    where the write-up runs it; float32 everywhere else, including T4 and P100 (7.5 and 6.0),
    which would emulate bf16, and CPU and MPS. The same line ``--amp bf16`` draws.

    Args:
        device: The parameter's device.

    Returns:
        ``torch.bfloat16`` or ``torch.float32``.
    """
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8:
        return torch.bfloat16
    return torch.float32


def update_scale(rows: int, cols: int, scale: str) -> float:
    """The factor an orthogonalised ``rows x cols`` update is multiplied by.

    Args:
        rows: The matrix's first dimension (a linear layer's output width).
        cols: Its second dimension.
        scale: ``"rms_match"`` (``0.2·sqrt(max(rows, cols))``) or ``"keller"``
            (``sqrt(max(1, rows/cols))``).

    Returns:
        The multiplier.

    Raises:
        ValueError: For an unknown ``scale``.
    """
    if scale == "rms_match":
        return _RMS_MATCH_TARGET * max(rows, cols) ** 0.5
    if scale == "keller":
        return max(1.0, rows / cols) ** 0.5
    raise ValueError(f"muon_scale must be one of {MUON_SCALES}, got {scale!r}")


def newton_schulz(
    g: torch.Tensor, steps: int = 5, eps: float = 1e-7, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Approximately orthogonalise a matrix: ``U S V^T`` to roughly ``U V^T``.

    The iteration is a contraction toward the orthogonal factor and tolerates bf16, which is
    what makes it cheap on GPUs that have bf16; :class:`Muon` picks ``dtype`` with
    :func:`newton_schulz_dtype`.

    Args:
        g: 2-D update matrix.
        steps: Newton-Schulz iterations.
        eps: Guard for the Frobenius normalisation of an all-zero update.
        dtype: Precision the iteration runs in.

    Returns:
        A ``dtype`` matrix of ``g``'s shape whose singular values are all near 1.
    """
    if g.ndim != 2:
        raise ValueError(f"newton_schulz needs a 2-D matrix, got shape {tuple(g.shape)}")
    a, b, c = _NS_COEFFS
    x = g.to(dtype)
    transposed = x.shape[0] > x.shape[1]
    if transposed:  # iterate on the wide orientation, so X X^T is the smaller Gram matrix
        x = x.T
    x = x / (x.norm() + eps)  # singular values into [0, 1], where the iteration converges
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x.T if transposed else x


class Muon(torch.optim.Optimizer):
    """Muon on groups flagged ``use_muon=True``, decoupled-weight-decay AdamW on the rest.

    Every group carries its own ``lr`` and ``weight_decay``. Muon groups also read
    ``momentum``, ``nesterov``, ``ns_steps``, ``scale`` and ``row_blocks``; AdamW groups read
    ``betas`` and ``eps``. Muon groups must hold only 2-D parameters whose row count
    ``row_blocks`` divides.

    Args:
        params: Parameter groups (dicts with ``params`` and ``use_muon``).
        lr: Default rate for groups that do not set one.
        momentum: Muon's SGD momentum (write-up default 0.95).
        nesterov: Use the Nesterov form of the momentum, as the write-up does.
        ns_steps: Newton-Schulz iterations per update.
        scale: Default update-scaling rule, one of :data:`MUON_SCALES`.
        row_blocks: Default number of equal row blocks a Muon matrix is orthogonalised as,
            each separately: 3 for a fused QKV projection, 1 otherwise.
        betas: AdamW's moment coefficients.
        eps: AdamW's denominator guard.
        weight_decay: Default decoupled weight decay.
    """

    def __init__(
        self,
        params: Iterable[dict[str, Any]],
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        scale: str = "rms_match",
        row_blocks: int = 1,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        """See the class docstring."""
        defaults = {
            "lr": lr, "momentum": momentum, "nesterov": nesterov, "ns_steps": ns_steps,
            "scale": scale, "row_blocks": row_blocks, "betas": betas, "eps": eps,
            "weight_decay": weight_decay, "use_muon": False,
        }
        super().__init__(params, defaults)
        #: Newton-Schulz precision per device, looked up once (not optimiser state).
        self._ns_dtypes: dict[torch.device, torch.dtype] = {}
        for group in self.param_groups:
            if not group["use_muon"]:
                continue
            if any(p.ndim != 2 for p in group["params"]):
                raise ValueError("Muon groups take 2-D weight matrices only")
            update_scale(1, 1, group["scale"])  # refuse an unknown rule now, not at step 1
            if any(p.shape[0] % group["row_blocks"] for p in group["params"]):
                raise ValueError(
                    f"row_blocks={group['row_blocks']} must divide every Muon matrix's rows"
                )

    def _orthogonalised(self, m: torch.Tensor, group: dict[str, Any]) -> torch.Tensor:
        """The scaled, orthogonalised update for one Muon matrix's momentum ``m``.

        Args:
            m: The (Nesterov) momentum, in the matrix's shape.
            group: The matrix's parameter group.

        Returns:
            The update in ``m``'s dtype, each row block orthogonalised and scaled by its own
            shape.
        """
        if m.device not in self._ns_dtypes:
            self._ns_dtypes[m.device] = newton_schulz_dtype(m.device)
        dtype = self._ns_dtypes[m.device]
        return torch.cat([
            newton_schulz(b, group["ns_steps"], dtype=dtype).to(m.dtype)
            * update_scale(b.shape[0], b.shape[1], group["scale"])
            for b in m.chunk(group["row_blocks"], dim=0)
        ])

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # type: ignore[override]
        """Apply one update to every parameter that has a gradient.

        Args:
            closure: Optional loss re-evaluation, per the ``torch.optim`` protocol.

        Returns:
            The closure's loss, or ``None``.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if wd:
                    p.mul_(1.0 - lr * wd)
                if group["use_muon"]:
                    if not state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.lerp_(g, 1.0 - group["momentum"])
                    update = g.lerp(buf, group["momentum"]) if group["nesterov"] else buf
                    p.add_(self._orthogonalised(update, group).to(p.dtype), alpha=-lr)
                else:
                    if not state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    beta1, beta2 = group["betas"]
                    state["step"] += 1
                    state["exp_avg"].lerp_(g, 1.0 - beta1)
                    state["exp_avg_sq"].mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                    bias1 = 1.0 - beta1 ** state["step"]
                    bias2 = 1.0 - beta2 ** state["step"]
                    denom = (state["exp_avg_sq"] / bias2).sqrt_().add_(group["eps"])
                    p.addcdiv_(state["exp_avg"], denom, value=-lr / bias1)
        return loss
