"""Muon, an opt-in pretraining optimiser (audit AU5.17), with an AdamW side for the rest.

Muon (Jordan et al., 2024, "Muon: An optimizer for hidden layers in neural networks",
kellerjordan.github.io/posts/muon; reference implementation MIT-licensed) takes SGD-momentum's
update for a 2-D weight matrix and replaces it with the nearest semi-orthogonal matrix,
computed by a few Newton-Schulz iterations. Written here from that description; nothing is
taken from any tabular-foundation-model project.

The write-up is explicit that Muon is for **hidden** weight matrices only: input embeddings,
the output head, and every gain or bias belong to a standard optimiser. So one
:class:`Muon` instance carries both kinds of parameter group, flagged by ``use_muon``, and
runs AdamW on the second kind. One optimiser object rather than two keeps the training loop's
single ``opt``/``sched``/``state_dict`` path unchanged: ``LambdaLR`` scales every group's rate
by the same warmup-and-cosine multiplier, and a resume restores both halves from one dict.

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


def newton_schulz(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximately orthogonalise a matrix: ``U S V^T`` to roughly ``U V^T``.

    Runs in bfloat16, as the write-up does: the iteration is a contraction toward the
    orthogonal factor and tolerates low precision, which is what makes it cheap.

    Args:
        g: 2-D update matrix.
        steps: Newton-Schulz iterations.
        eps: Guard for the Frobenius normalisation of an all-zero update.

    Returns:
        A bfloat16 matrix of ``g``'s shape whose singular values are all near 1.
    """
    if g.ndim != 2:
        raise ValueError(f"newton_schulz needs a 2-D matrix, got shape {tuple(g.shape)}")
    a, b, c = _NS_COEFFS
    x = g.bfloat16()
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
    ``momentum``, ``nesterov`` and ``ns_steps``; AdamW groups read ``betas`` and ``eps``.
    Muon groups must hold only 2-D parameters.

    Args:
        params: Parameter groups (dicts with ``params`` and ``use_muon``).
        lr: Default rate for groups that do not set one.
        momentum: Muon's SGD momentum (write-up default 0.95).
        nesterov: Use the Nesterov form of the momentum, as the write-up does.
        ns_steps: Newton-Schulz iterations per update.
        betas: AdamW's moment coefficients.
        eps: AdamW's denominator guard.
        weight_decay: Default decoupled weight decay.
    """

    def __init__(
        self,
        params: Iterable[dict[str, Any]],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        """See the class docstring."""
        defaults = {
            "lr": lr, "momentum": momentum, "nesterov": nesterov, "ns_steps": ns_steps,
            "betas": betas, "eps": eps, "weight_decay": weight_decay, "use_muon": False,
        }
        super().__init__(params, defaults)
        for group in self.param_groups:
            if group["use_muon"] and any(p.ndim != 2 for p in group["params"]):
                raise ValueError("Muon groups take 2-D weight matrices only")

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
                    update = newton_schulz(update, group["ns_steps"])
                    # an orthogonal update's RMS shrinks with a tall matrix's aspect ratio;
                    # the write-up's correction restores it
                    scale = max(1.0, p.shape[0] / p.shape[1]) ** 0.5
                    p.add_(update.to(p.dtype), alpha=-lr * scale)
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
