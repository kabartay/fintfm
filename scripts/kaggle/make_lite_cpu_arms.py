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
#: The fork pin of the 2026-11-16 arms; an arm may name another (branch, commit) in ``FORKS``.
DEFAULT_FORK = ("fintfm-lite-arms-1116", "b8612d7")
#: Arms on another fork commit, with the model tests run in the kernel before fitting.
FORKS: dict[str, tuple[str, str]] = {
    "cpu_mc_onehot": ("fintfm-mc-onehot", "229af56"),
    "cpu_mc_onehot_floor": ("fintfm-audit-fixes", "a52725c"),
    "cpu_mc_spline_small": ("fintfm-audit-fixes", "bc99785"),
    "cpu_reg_spline_small": ("fintfm-audit-fixes", "bc99785"),
}
#: Kernel directory -> (run name, problem type, lever variables).
ARMS: dict[str, tuple[str, str, dict[str, str]]] = {
    "cpu_mc_ctrl": ("fintfm_lp_mc_ctrl_cpu", "multiclass", {}),
    "cpu_mc_floor": ("fintfm_lp_mc_floor_cpu", "multiclass", {"FINTFM_PROB_FLOOR": "cv"}),
    "cpu_reg_ctrl": ("fintfm_lp_reg_ctrl_cpu", "regression", {}),
    "cpu_reg_rff": ("fintfm_lp_reg_rff_cpu", "regression", {"FINTFM_REG_READOUT": "rff"}),
    "cpu_mc_onehot": ("fintfm_lp_mc_onehot_cpu", "multiclass", {"FINTFM_PROBE_ONEHOT": "on"}),
    "cpu_mc_onehot_floor": (
        "fintfm_lp_mc_onehot_floor_cpu",
        "multiclass",
        {"FINTFM_PROBE_ONEHOT": "on", "FINTFM_PROB_FLOOR": "cv"},
    ),
    # FINDINGS §292: the entry's settings for the problem type plus spline features.
    "cpu_mc_spline_small": (
        "fintfm_lp_mc_spline_small_cpu",
        "multiclass",
        {
            "FINTFM_PROBE_ONEHOT": '{"min_count": 3, "max_columns": 256, "max_levels": 64}',
            "FINTFM_PROB_FLOOR": "cv",
            "FINTFM_PROBE_SPLINE": "on",
        },
    ),
    "cpu_reg_spline_small": (
        "fintfm_lp_reg_spline_small_cpu",
        "regression",
        {
            "FINTFM_REG_READOUT": "rff",
            "FINTFM_REG_RFF_MAX_ROWS": "5000",
            "FINTFM_PROBE_SPLINE": "on",
        },
    ),
}
#: Arms that run a subset of their problem type's tables (FINDINGS §292); the others run all.
ONLY: dict[str, list[str]] = {
    "cpu_mc_spline_small": [
        "anneal",
        "hiva_agnostic",
        "maternal_health_risk",
        "MIC",
        "splice",
        "students_dropout_and_academic_success",
        "website_phishing",
    ],
    "cpu_reg_spline_small": [
        "QSAR_fish_toxicity",
        "concrete_compressive_strength",
        "healthcare_insurance_expenses",
        "airfoil_self_noise",
        "Another-Dataset-on-used-Fiat-500",
        "QSAR-TID-11",
        "wine_quality",
    ],
}
MARK = "# --- ARM ---\n"


def render(run: str, problem: str, lever: dict[str, str], name: str = "") -> str:
    """The template with its arm block replaced.

    Args:
        run: The run name.
        problem: ``multiclass`` or ``regression``.
        lever: Extra environment variables of the arm.
        name: The kernel directory key, which selects a fork pin from ``FORKS``.

    Returns:
        The kernel's source.
    """
    branch, commit = FORKS.get(name, DEFAULT_FORK)
    head, _, rest = TEMPLATE.read_text().partition(MARK)
    _, _, tail = rest.partition(MARK)
    only = ONLY.get(name)
    block = (
        f'RUN_NAME = "{run}"\nPROBLEM_TYPE = "{problem}"\n'
        f"LEVER: dict[str, str] = {json.dumps(lever)}\n"
        f"DATASETS_ONLY: list[str] | None = {json.dumps(only) if only else None}\n"
        f'FORK_COMMIT = "{commit}"\nFORK_BRANCH = "{branch}"\n'
        f"RUN_MODEL_TESTS = {name in FORKS}\n"
    )
    return head + MARK + block + MARK + tail


def main() -> None:
    """Write every kernel directory."""
    for name, (run, problem, lever) in ARMS.items():
        d = HERE / f"task_lite_{name}"
        d.mkdir(exist_ok=True)
        code = f"task_lite_{name}.py"
        (d / code).write_text(render(run, problem, lever, name))
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
