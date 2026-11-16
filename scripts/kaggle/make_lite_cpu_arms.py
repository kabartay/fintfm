"""Generate the Kaggle CPU kernels for the 2026-11-16 multiclass and regression Lite arms.

Each kernel is ``lite_cpu_arm_template.py`` with its ``# --- ARM ---`` block filled, plus a
``kernel-metadata.json`` (private, CPU, internet, the checkpoint's training kernel as source).
Run ``python scripts/kaggle/make_lite_cpu_arms.py`` after editing the template.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "lite_cpu_arm_template.py"
SOURCES = {
    "multiclass": "muhakabartay/fintfm-ts-dr74-mc-s0",
    "regression": "muhakabartay/fintfm-task-48-3-matched-binned-head-control",
}
#: Kernel directory -> (run name, problem type, lever variables).
ARMS: dict[str, tuple[str, str, dict[str, str]]] = {
    "cpu_mc_ctrl": ("fintfm_lp_mc_ctrl_cpu", "multiclass", {}),
    "cpu_mc_floor": ("fintfm_lp_mc_floor_cpu", "multiclass", {"FINTFM_PROB_FLOOR": "cv"}),
    "cpu_reg_ctrl": ("fintfm_lp_reg_ctrl_cpu", "regression", {}),
    "cpu_reg_rff": ("fintfm_lp_reg_rff_cpu", "regression", {"FINTFM_REG_READOUT": "rff"}),
}
MARK = "# --- ARM ---\n"


def render(run: str, problem: str, lever: dict[str, str]) -> str:
    """The template with its arm block replaced.

    Args:
        run: The run name.
        problem: ``multiclass`` or ``regression``.
        lever: Extra environment variables of the arm.

    Returns:
        The kernel's source.
    """
    head, _, rest = TEMPLATE.read_text().partition(MARK)
    _, _, tail = rest.partition(MARK)
    block = (
        f'RUN_NAME = "{run}"\nPROBLEM_TYPE = "{problem}"\n'
        f"LEVER: dict[str, str] = {json.dumps(lever)}\n"
    )
    return head + MARK + block + MARK + tail


def main() -> None:
    """Write every kernel directory."""
    for name, (run, problem, lever) in ARMS.items():
        d = HERE / f"task_lite_{name}"
        d.mkdir(exist_ok=True)
        code = f"task_lite_{name}.py"
        (d / code).write_text(render(run, problem, lever))
        slug = f"fintfm-task-lite-{name.replace('_', '-')}"
        meta = {
            "id": f"muhakabartay/{slug}",
            "title": slug,
            "code_file": code,
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "false",
            "enable_tpu": "false",
            "enable_internet": "true",
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [SOURCES[problem]],
            "model_sources": [],
        }
        (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"wrote {d}")


if __name__ == "__main__":
    main()
