"""Scaling-curve grid: sweep pretraining volume, model size and seed as one config.

Task 9.3 of ``openspec/changes/scaling-curve``. The prior ablation varies the prior mixture
at one scale; this module varies *scale* (tasks seen and parameter count) on one recipe. A
YAML spec under ``configs/scaling/`` names the recipe, the task counts, the model sizes and
the seeds per task count; :func:`expand_grid` turns it into cells, :func:`plan_grid` prints
each cell's training command and ESTIMATED cost without training anything, and
:func:`run_cell` trains one cell and writes its own ``results.json``.

Volume is counted in tasks, not steps (CLAUDE.md): ``tasks_seen = steps x batch_size``, and a
cell whose task count is not a multiple of the batch size is refused rather than rounded.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shlex
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from fintfm.config import load_config
from fintfm.evaluation.coverage import Coverage
from fintfm.experiments.prior_ablation import _git_commit, evaluate_on_credit
from fintfm.modeling.model import FinancialTFM
from fintfm.modeling.train import build_configs, build_parser, train

_SPEC_KEYS = {
    "name",
    "recipe",
    "batch_size",
    "device",
    "checkpoint_every",
    "horizons",
    "tasks",
    "seeds",
    "models",
    "cost",
}
#: Model fields a size entry may override, mapped to their ``fintfm-train`` flag.
_MODEL_FLAGS = {
    "d_cell": "--d-cell",
    "d_model": "--d-model",
    "n_layers": "--n-layers",
    "n_col_layers": "--n-col-layers",
    "n_heads": "--n-heads",
    "d_ff": "--d-ff",
}
_COST_KEYS = {"gpus_per_session", "quota_hours_per_week"}
_SIZE_KEYS = set(_MODEL_FLAGS) | {"step_seconds", "step_seconds_basis"}


@dataclass(frozen=True)
class GridSpec:
    """A parsed grid specification; see ``configs/scaling/grid-2026-11-15.yaml``.

    Attributes:
        name: Label recorded in every cell's results.
        recipe: Path of the training recipe every cell starts from.
        batch_size: Tasks per step, identical across cells.
        device: Training device written into each command.
        checkpoint_every: Steps between checkpoints; must be positive.
        horizons: Bankruptcy horizons scored after training; empty trains only.
        tasks: Pretraining volumes, in tasks.
        seeds: Seeds to run at each task count.
        models: Size label to its flag overrides, ``step_seconds`` and its basis.
        gpus_per_session: GPUs billed per wall hour.
        quota_hours_per_week: Weekly quota used to express the total in weeks.
    """

    name: str
    recipe: str
    batch_size: int
    device: str
    checkpoint_every: int
    horizons: tuple[int, ...]
    tasks: tuple[int, ...]
    seeds: dict[int, tuple[int, ...]]
    models: dict[str, dict[str, Any]]
    gpus_per_session: int = 2
    quota_hours_per_week: float = 30.0


@dataclass(frozen=True)
class Cell:
    """One (task count, model size, seed) training run.

    Attributes:
        cell_id: Directory-safe identifier, ``<size>-t<tasks>-s<seed>``.
        size: Model size label from the spec.
        tasks: Tasks seen, equal to ``steps * batch_size``.
        steps: Gradient steps.
        batch_size: Tasks per step.
        seed: Training seed.
        overrides: Model flag overrides applied on the recipe.
    """

    cell_id: str
    size: str
    tasks: int
    steps: int
    batch_size: int
    seed: int
    overrides: dict[str, int] = field(default_factory=dict)

    @property
    def tasks_seen(self) -> int:
        """Tasks the model sees: ``steps x batch_size``."""
        return self.steps * self.batch_size


def load_grid(path: str | Path) -> GridSpec:
    """Read and validate a grid spec.

    Args:
        path: YAML file.

    Returns:
        The spec.

    Raises:
        ValueError: On an unknown key, a task count that is not a multiple of the batch size,
            a task count with no seeds, or a non-positive ``checkpoint_every``.
    """
    data = yaml.safe_load(Path(path).read_text())
    unknown = sorted(set(data) - _SPEC_KEYS)
    if unknown:
        raise ValueError(f"unknown grid key(s) {unknown}; valid keys are {sorted(_SPEC_KEYS)}")
    cost = data.get("cost", {})
    bad_cost = sorted(set(cost) - _COST_KEYS)
    if bad_cost:
        raise ValueError(f"unknown cost key(s) {bad_cost}")
    models = data["models"]
    for label, entry in models.items():
        bad = sorted(set(entry) - _SIZE_KEYS)
        if bad:
            raise ValueError(f"unknown key(s) {bad} in model size {label!r}")
        if "step_seconds" not in entry:
            raise ValueError(f"model size {label!r} needs step_seconds for the cost estimate")
    batch = int(data["batch_size"])
    every = int(data["checkpoint_every"])
    if every <= 0:
        raise ValueError("checkpoint_every must be positive: every launch carries it")
    tasks = tuple(int(t) for t in data["tasks"])
    seeds = {int(k): tuple(int(s) for s in v) for k, v in data["seeds"].items()}
    for t in tasks:
        if t % batch:
            raise ValueError(f"{t} tasks is not a multiple of batch_size {batch}")
        if not seeds.get(t):
            raise ValueError(f"no seeds given for {t} tasks")
    return GridSpec(
        name=data["name"],
        recipe=data["recipe"],
        batch_size=batch,
        device=data["device"],
        checkpoint_every=every,
        horizons=tuple(int(h) for h in data.get("horizons", ())),
        tasks=tasks,
        seeds=seeds,
        models=models,
        gpus_per_session=int(cost.get("gpus_per_session", 2)),
        quota_hours_per_week=float(cost.get("quota_hours_per_week", 30.0)),
    )


def expand_grid(spec: GridSpec) -> list[Cell]:
    """Expand a spec into its cells: every size x task count x that task count's seeds.

    Args:
        spec: A validated spec.

    Returns:
        Cells ordered by size, then tasks, then seed.
    """
    cells = []
    for size, entry in spec.models.items():
        overrides = {k: int(v) for k, v in entry.items() if k in _MODEL_FLAGS}
        if "d_model" in overrides and "d_ff" not in overrides:
            overrides["d_ff"] = 4 * overrides["d_model"]  # the transformer convention
        for tasks in spec.tasks:
            for seed in spec.seeds[tasks]:
                cells.append(
                    Cell(
                        cell_id=f"{size}-t{tasks}-s{seed}",
                        size=size,
                        tasks=tasks,
                        steps=tasks // spec.batch_size,
                        batch_size=spec.batch_size,
                        seed=seed,
                        overrides=overrides,
                    )
                )
    return cells


def cell_argv(spec: GridSpec, cell: Cell, out: str) -> list[str]:
    """The ``fintfm-train`` arguments (without the program name) that train one cell.

    Args:
        spec: The grid spec.
        cell: The cell.
        out: Checkpoint path.

    Returns:
        Argument list, always including ``--checkpoint-every``.
    """
    argv = [
        "--config", spec.recipe,
        "--steps", str(cell.steps),
        "--batch-size", str(cell.batch_size),
        "--seed", str(cell.seed),
        "--device", spec.device,
        "--checkpoint-every", str(spec.checkpoint_every),
    ]  # fmt: skip
    for key, value in cell.overrides.items():
        argv += [_MODEL_FLAGS[key], str(value)]
    return argv + ["--out", out]


def cell_command(spec: GridSpec, cell: Cell, out_root: Path) -> str:
    """The shell command that trains one cell.

    Args:
        spec: The grid spec.
        cell: The cell.
        out_root: Directory holding one sub-directory per cell.

    Returns:
        A ``uv run fintfm-train ...`` line.
    """
    out = str(out_root / cell.cell_id / "model.pt")
    return "uv run fintfm-train " + shlex.join(cell_argv(spec, cell, out))


def _parameter_count(spec: GridSpec, cell: Cell) -> int:
    """Parameter count of a cell's model, from constructing it (no training happens)."""
    args = build_parser().parse_args(cell_argv(spec, cell, "unused.pt"))
    model_cfg, _, _ = build_configs(args, load_config(spec.recipe))
    return FinancialTFM(model_cfg).num_parameters()


def plan_grid(spec: GridSpec, out_root: Path) -> str:
    """Describe every cell's command and ESTIMATED cost; trains nothing.

    Args:
        spec: The grid spec.
        out_root: Directory the cells would write under.

    Returns:
        The plan as text. Every cost is ESTIMATED from the spec's ``step_seconds``, which
        are labelled measured or estimated per size.
    """
    lines = [f"PLAN {spec.name}: nothing is trained; every cost is ESTIMATED", ""]
    total_wall = 0.0
    cells = expand_grid(spec)
    for cell in cells:
        entry = spec.models[cell.size]
        wall_h = cell.steps * float(entry["step_seconds"]) / 3600
        total_wall += wall_h
        basis = entry.get("step_seconds_basis", "estimated")
        lines += [
            (
                f"[{cell.cell_id}] {_parameter_count(spec, cell):,} parameters, "
                f"{cell.steps:,} steps x batch {cell.batch_size} = {cell.tasks_seen:,} tasks seen"
            ),
            (
                f"  ESTIMATED {wall_h:.2f} wall h ({wall_h * spec.gpus_per_session:.2f} GPU-h) "
                f"at {entry['step_seconds']} s/step ({basis} step time)"
            ),
            f"  {cell_command(spec, cell, out_root)}",
        ]
    gpu_h = total_wall * spec.gpus_per_session
    lines += [
        "",
        (
            f"{len(cells)} cells: ESTIMATED {total_wall:.1f} wall h, {gpu_h:.1f} GPU-h, "
            f"{total_wall / spec.quota_hours_per_week:.1f} weeks at "
            f"{spec.quota_hours_per_week:g} quota h"
        ),
    ]
    return "\n".join(lines)


def run_cell(spec: GridSpec, cell: Cell, out_root: Path) -> dict:
    """Train one cell, score it, and write ``<out_root>/<cell_id>/results.json``.

    Args:
        spec: The grid spec.
        cell: The cell to run.
        out_root: Directory holding one sub-directory per cell.

    Returns:
        The record written: commit, recipe, steps, batch size, tasks seen, parameter count.
    """
    out_dir = out_root / cell.cell_id
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "model.pt"
    argv = cell_argv(spec, cell, str(ckpt))
    args = build_parser().parse_args(argv)
    model_cfg, prior_cfg, train_cfg = build_configs(args, load_config(spec.recipe))
    started = time.time()
    model = train(model_cfg, prior_cfg, train_cfg, str(ckpt))
    elapsed = time.time() - started
    cov = Coverage()
    metrics = (
        evaluate_on_credit(str(ckpt), horizons=spec.horizons, seed=cell.seed, coverage=cov)
        if spec.horizons
        else {}
    )
    record = {
        "created": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "platform": platform.platform(),
        "grid": spec.name,
        "cell": cell.cell_id,
        "recipe": {
            "path": spec.recipe,
            "sha256": hashlib.sha256(Path(spec.recipe).read_bytes()).hexdigest(),
            "model_overrides": cell.overrides,
        },
        "command": "uv run fintfm-train " + shlex.join(argv),
        "seed": cell.seed,
        "steps": train_cfg.steps,
        "batch_size": train_cfg.batch_size,
        "tasks_seen": train_cfg.steps * train_cfg.batch_size,
        "n_parameters": model.num_parameters(),
        "model": asdict(model_cfg),
        "checkpoint_every": train_cfg.checkpoint_every,
        "device": train_cfg.device,
        "torch_threads": torch.get_num_threads(),
        "train_seconds": elapsed,
        "horizons": list(spec.horizons),
        "metrics": metrics,
        "coverage": cov.to_dict() if spec.horizons else None,
    }
    (out_dir / "results.json").write_text(json.dumps(record, indent=2))
    print(f"wrote {out_dir / 'results.json'}")
    return record


def run_grid(spec: GridSpec, out_root: Path) -> list[dict]:
    """Train every cell in turn, one ``results.json`` each.

    Args:
        spec: The grid spec.
        out_root: Directory holding one sub-directory per cell.

    Returns:
        The records, in cell order.
    """
    return [run_cell(spec, cell, out_root) for cell in expand_grid(spec)]
