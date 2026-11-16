"""Generate the Kaggle CPU kernels that run openspec 37.3 and 37.4 (``suite_scoring``), 2026-11-16.

Each kernel is ``cpu_37_suite_template.py`` with its ``# --- PART ---`` block filled, plus a
``kernel-metadata.json`` (private, CPU, internet, no kernel or dataset sources: every input is a public
download verified by SHA-256). The datasets are split so each part's estimated time is well inside
Kaggle's 12 h limit. Run ``python scripts/kaggle/make_cpu_37_kernels.py`` after editing the template.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "cpu_37_suite_template.py"
#: Part letter -> datasets, balanced on rows x features (the cost measure of the time estimate).
PARTS: dict[str, list[str]] = {
    "a": ["clf_cat/covertype", "clf_num/bank-marketing"],
    "b": ["clf_num/MiniBooNE", "clf_num/MagicTelescope"],
    "c": ["clf_cat/road-safety", "clf_num/Higgs"],
    "d": [
        "clf_num/covertype",
        "clf_cat/default-of-credit-card-clients",
        "clf_num/default-of-credit-card-clients",
        "clf_num/Diabetes130US",
    ],
}
MARK = "# --- PART ---\n"


def render(part: str, datasets: list[str]) -> str:
    """The template with its part block replaced.

    Args:
        part: Part letter.
        datasets: Cleared dataset keys this kernel scores, in order.

    Returns:
        The kernel's source.
    """
    head, _, rest = TEMPLATE.read_text().partition(MARK)
    _, _, tail = rest.partition(MARK)
    block = f'PART = "{part}"\nDATASETS: list[str] = {json.dumps(datasets)}\n'
    return head + MARK + block + MARK + tail


def main() -> None:
    """Write every kernel directory."""
    for part, datasets in PARTS.items():
        slug = f"fintfm-task-cpu-37-suite-{part}"
        d = HERE / f"task_cpu_37_suite_{part}"
        d.mkdir(exist_ok=True)
        code = f"task_cpu_37_suite_{part}.py"
        (d / code).write_text(render(part, datasets))
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
            "kernel_sources": [],
            "model_sources": [],
        }
        (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"wrote {d}")


if __name__ == "__main__":
    main()
