"""Low-rank adapters over a frozen :class:`~fintfm.modeling.model.FinancialTFM` (task 16.2).

Why this exists
---------------
``openspec/changes/lender-adaptation`` asks whether a lender can specialise the shared base
model cheaply, without retraining it. This module is the mechanism: a LoRA-style update
``W + (alpha / rank) * B @ A`` on selected weight matrices, with ``B`` initialised to zero so
an adapter that has not been trained computes exactly what the base computes.

How it is attached
------------------
The model reads its attention weights directly (``mha.in_proj_weight``, ``mha.out_proj``)
rather than by calling the module, so wrapping a module would be bypassed. The update is
therefore a :mod:`torch.nn.utils.parametrize` parametrisation of the weight tensor itself,
which every read of ``module.weight`` sees. No forward code in ``model.py`` changes, and a
model with no adapter attached is the unmodified base.

Rules (``openspec/specs/pretraining-provenance`` P5)
----------------------------------------------------
* The base stays frozen: every base parameter has ``requires_grad=False`` once an adapter is
  attached, and only the adapter's ``A``/``B`` matrices train.
* An adapter is a **separate artefact**: it is saved on its own with a provenance record that
  says it was fitted on real data (``fitted_on_real_data: true``), so it inherits the
  FinTFM-R side of the provenance table, never the synthetic-only claim.
* An adapter is **never merged** into a base: :func:`merge_adapter_into_base` always refuses,
  and ``FinancialTFM.save`` refuses a model that carries one.

Always attach to a deep copy of the base (the classifier does); attaching modifies the model
it is given.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.utils import parametrize

from fintfm.modeling.model import FinancialTFM

#: Bumped on any incompatible change to the adapter file's layout.
ADAPTER_FORMAT = 1

#: Groups of weight matrices an adapter may target. ``attention``: the packed input
#: projection and the output projection of every ``nn.MultiheadAttention``. ``linear``: every
#: other ``nn.Linear`` outside the head (value embedding, row projection, feed-forward).
#: ``head``: the output head.
VALID_TARGETS: tuple[str, ...] = ("attention", "linear", "head")


class AdapterError(RuntimeError):
    """An adapter was used in a way P5 forbids, or does not belong to this base."""


@dataclass(frozen=True)
class AdapterConfig:
    """Shape of a low-rank adapter and of the few steps that fit it.

    Attributes:
        rank: Rank ``r`` of each ``B @ A`` update; at least 1.
        alpha: Scale numerator; the update is multiplied by ``alpha / rank``. ``None`` means
            ``alpha = rank``, a scale of 1.
        targets: Weight groups to adapt, a non-empty subset of :data:`VALID_TARGETS`.
        steps: Gradient steps taken by the classifier's ``fit_adapter``.
        lr: AdamW learning rate for the adapter parameters.
        batch: Tasks per step.
        query: Query rows per task, whose labels the loss is taken on.
        context: Context rows per task; ``None`` uses the classifier's ``max_context``.
        query_group_frac: Fraction of groups held out as a task's queries.
        seed: Seeds the adapter's ``A`` initialisation and the task draws.
        clip: Gradient-norm clip per step, or ``None`` for none.
    """

    rank: int = 4
    alpha: float | None = None
    targets: tuple[str, ...] = ("attention",)
    steps: int = 50
    lr: float = 1e-3
    batch: int = 2
    query: int = 512
    context: int | None = None
    query_group_frac: float = 0.3
    seed: int = 0
    clip: float | None = None

    def __post_init__(self) -> None:
        """Reject settings that cannot describe an adapter."""
        if self.rank < 1:
            raise ValueError(f"rank must be at least 1, got {self.rank}")
        if self.alpha is not None and self.alpha <= 0:
            raise ValueError(f"alpha must be None or positive, got {self.alpha}")
        if not self.targets or any(t not in VALID_TARGETS for t in self.targets):
            raise ValueError(f"targets must be a non-empty subset of {VALID_TARGETS}")
        if self.steps < 1 or self.batch < 1 or self.query < 1:
            raise ValueError("steps, batch and query must be at least 1")
        if self.context is not None and self.context < 1:
            raise ValueError(f"context must be None or >= 1, got {self.context}")
        if not 0.0 < self.query_group_frac < 1.0:
            raise ValueError("query_group_frac must lie strictly between 0 and 1")
        if self.clip is not None and self.clip <= 0:
            raise ValueError("clip must be None or positive")

    @property
    def scale(self) -> float:
        """The multiplier ``alpha / rank`` applied to every low-rank update."""
        return (self.rank if self.alpha is None else self.alpha) / self.rank


class LowRankUpdate(nn.Module):
    """Parametrisation ``W -> W + scale * B @ A`` with ``B`` zero-initialised.

    Args:
        out_features: Rows of the weight matrix.
        in_features: Columns of the weight matrix.
        rank: Rank of the update.
        scale: Multiplier on the update.
        generator: Seeds ``A``'s initialisation.
    """

    def __init__(
        self,
        out_features: int,
        in_features: int,
        rank: int,
        scale: float,
        generator: torch.Generator,
    ) -> None:
        super().__init__()
        self.scale = scale
        bound = 1.0 / math.sqrt(in_features)
        self.A = nn.Parameter((torch.rand(rank, in_features, generator=generator) * 2 - 1) * bound)
        self.B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        """Return the adapted weight.

        Args:
            weight: The frozen base weight, ``(out_features, in_features)``.

        Returns:
            ``weight`` plus the scaled low-rank update; exactly ``weight`` while ``B`` is 0.
        """
        return weight + self.scale * (self.B @ self.A)


def _target_weights(model: nn.Module, targets: tuple[str, ...]) -> list[tuple[str, nn.Module, str]]:
    """The ``(qualified name, module, attribute)`` triples an adapter over ``targets`` covers.

    Args:
        model: The model to scan.
        targets: Subset of :data:`VALID_TARGETS`.

    Returns:
        Triples in module order, so the adapter's parameter names are stable across loads.
    """
    out: list[tuple[str, nn.Module, str]] = []
    in_mha: set[int] = set()
    for name, mod in model.named_modules():
        if isinstance(mod, nn.MultiheadAttention):
            in_mha.add(id(mod.out_proj))
            if "attention" in targets and mod.in_proj_weight is not None:
                out.append((f"{name}.in_proj_weight", mod, "in_proj_weight"))
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if id(mod) in in_mha:
            if "attention" in targets:
                out.append((f"{name}.weight", mod, "weight"))
        elif name == "head" or name.startswith("head."):
            if "head" in targets:
                out.append((f"{name}.weight", mod, "weight"))
        elif "linear" in targets:
            out.append((f"{name}.weight", mod, "weight"))
    return out


def attach_adapter(model: FinancialTFM, config: AdapterConfig) -> list[str]:
    """Freeze ``model`` and attach a zero-initialised low-rank adapter, in place.

    Pass a deep copy of the base: the model is modified.

    Args:
        model: The base model, or a copy of it.
        config: Rank, scale and targets.

    Returns:
        The qualified weight names adapted, in the order their parameters are stored.

    Raises:
        AdapterError: If the model already carries an adapter.
        ValueError: If ``config.targets`` selects no weight in this model.
    """
    if getattr(model, "adapter_config", None) is not None:
        raise AdapterError("this model already carries an adapter; attach to a fresh copy")
    triples = _target_weights(model, config.targets)
    if not triples:
        raise ValueError(f"targets {config.targets} select no weight in this model")
    gen = torch.Generator().manual_seed(config.seed)
    for p in model.parameters():
        p.requires_grad_(False)
    for _, mod, attr in triples:
        w = getattr(mod, attr)
        update = LowRankUpdate(w.shape[0], w.shape[1], config.rank, config.scale, gen)
        update.to(device=w.device, dtype=w.dtype)
        parametrize.register_parametrization(mod, attr, update)
    names = [n for n, _, _ in triples]
    model.adapter_config = config  # type: ignore[assignment]
    model.adapter_targets = names  # type: ignore[assignment]
    return names


def adapter_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    """The adapter's trainable ``A``/``B`` parameters, by qualified name.

    Args:
        model: A model with an adapter attached.

    Returns:
        ``{name: parameter}`` for exactly the parameters inside a :class:`LowRankUpdate`.
    """
    return {
        n: p
        for n, p in model.named_parameters()
        if n.rsplit(".", 1)[-1] in ("A", "B") and ".parametrizations." in n
    }


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Detached CPU copies of the adapter parameters, the contents of an adapter file."""
    return {n: p.detach().cpu().clone() for n, p in adapter_parameters(model).items()}


def load_adapter_state(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    """Copy saved adapter parameters into a model that has the same adapter attached.

    Args:
        model: Model with an adapter of identical rank and targets attached.
        state: Output of :func:`adapter_state_dict`.

    Raises:
        AdapterError: If the names or shapes differ from the attached adapter.
    """
    params = adapter_parameters(model)
    if set(params) != set(state):
        raise AdapterError("adapter file does not match the attached adapter's parameters")
    with torch.no_grad():
        for n, p in params.items():
            if tuple(p.shape) != tuple(state[n].shape):
                raise AdapterError(f"adapter parameter {n} has shape {tuple(state[n].shape)}")
            p.copy_(state[n].to(p.device, p.dtype))


def base_weights_sha256(model: nn.Module) -> str:
    """SHA-256 over the base weights' names, shapes and bytes, in sorted-name order.

    Adapter parametrisations are skipped, and the ``original`` tensors of adapted weights are
    mapped back to the base's own names, so a model's hash is the same before and after an
    adapter is attached: it identifies the base, which a file hash cannot do for an
    in-memory model.

    Args:
        model: A base model or an adapted one.

    Returns:
        The lowercase hexadecimal digest.
    """
    digest = hashlib.sha256()
    items: dict[str, torch.Tensor] = {}
    for n, t in model.state_dict().items():
        if ".parametrizations." in n:
            if not n.endswith(".original"):
                continue
            head, rest = n.split(".parametrizations.", 1)
            n = f"{head}.{rest.split('.', 1)[0]}"
        items[n] = t
    for n in sorted(items):
        t = items[n].detach().cpu().contiguous()
        digest.update(n.encode())
        digest.update(str(tuple(t.shape)).encode())
        digest.update(t.numpy().tobytes())
    return digest.hexdigest()


def merge_adapter_into_base(*_: Any, **__: Any) -> None:
    """Always refuse: an adapter is a per-lender artefact and is never merged (P5).

    Raises:
        AdapterError: Always. Merging one lender's adapter into a base that is shipped would
            carry their book into everyone else's predictions and void provenance P1.
    """
    raise AdapterError(
        "an adapter is never merged into a base model (provenance P5): keep it as a "
        "separate per-lender artefact and attach it at inference"
    )


def save_adapter(
    path: str | Path,
    model: nn.Module,
    provenance: Mapping[str, Any],
) -> None:
    """Write an adapter to its own file, with its provenance record.

    Args:
        path: Destination, never a base checkpoint's path.
        model: The adapted model.
        provenance: Record built by the classifier; stored beside the weights.

    Raises:
        AdapterError: If ``model`` carries no adapter, or the record does not state that
            real data was used (P5 makes that the one thing it must say).
    """
    from fintfm.provenance import json_safe

    cfg = getattr(model, "adapter_config", None)
    if cfg is None:
        raise AdapterError("this model carries no adapter to save")
    if provenance.get("fitted_on_real_data") is not True:
        raise AdapterError("an adapter's provenance must record fitted_on_real_data: true")
    torch.save(
        {
            "format": ADAPTER_FORMAT,
            "config": json_safe(asdict(cfg)),
            "targets": list(model.adapter_targets),  # type: ignore[arg-type]
            "state": adapter_state_dict(model),
            "provenance": json_safe(dict(provenance)),
        },
        str(path),
    )


def read_adapter(path: str | Path) -> dict[str, Any]:
    """Read an adapter file with ``weights_only=True``.

    Args:
        path: File written by :func:`save_adapter`.

    Returns:
        The stored dict: ``format``, ``config``, ``targets``, ``state``, ``provenance``.
    """
    return torch.load(str(path), map_location="cpu", weights_only=True)
