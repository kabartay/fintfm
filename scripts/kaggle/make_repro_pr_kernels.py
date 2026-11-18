"""Write the TabArena-Lite reproduction kernels from ``repro_pr_kernel_template.py``.

Run from the repository root: ``.venv/bin/python scripts/kaggle/make_repro_pr_kernels.py``. The
dataset lists are the Lite suite's 51 datasets (30 binary, 8 multiclass, 13 regression), the same
ones ``task_lite_air_levers`` and ``task_lite_xfit_mcreg`` use.
"""

from __future__ import annotations

import json
import pprint
from pathlib import Path

HERE = Path(__file__).parent

BINARY = [
    "Amazon_employee_access", "Bank_Customer_Churn", "Diabetes130US", "E-CommereShippingData",
    "Fitness_Club", "GiveMeSomeCredit", "HR_Analytics_Job_Change_of_Data_Scientists",
    "Is-this-a-good-customer", "Marketing_Campaign", "NATICUSdroid", "bank-marketing",
    "blood-transfusion-service-center", "churn", "coil2000_insurance_policies", "credit-g",
    "credit_card_clients_default", "diabetes", "hazelnut-spread-contaminant-detection", "heloc",
    "in_vehicle_coupon_recommendation", "jm1", "online_shoppers_intention",
    "polish_companies_bankruptcy", "qsar-biodeg", "seismic-bumps", "taiwanese_bankruptcy_prediction",
    "APSFailure", "Bioresponse", "kddcup09_appetency", "customer_satisfaction_in_airline",
]  # fmt: skip
MULTICLASS = [
    "anneal", "hiva_agnostic", "maternal_health_risk", "MIC", "SDSS17", "splice",
    "students_dropout_and_academic_success", "website_phishing",
]  # fmt: skip
REGRESSION = [
    "QSAR_fish_toxicity", "concrete_compressive_strength", "healthcare_insurance_expenses",
    "airfoil_self_noise", "Another-Dataset-on-used-Fiat-500", "QSAR-TID-11", "wine_quality",
    "miami_housing", "houses", "superconductivity", "Food_Delivery_Time", "physiochemical_protein",
    "diamonds",
]  # fmt: skip

#: The binary suite by fit size, matching the device each part of the §290 entry was measured on:
#: the 21 tables of at most 10,000 training rows ran on CPU (Kaggle and the Mac), so they run on
#: two Kaggle CPU kernels, balanced by stored fit time (about 3.6 h each); the 9 larger tables ran
#: on a Kaggle T4, so they keep one GPU kernel (2026-11-17).
BINARY_SMALL_A = [
    "Bioresponse",
    "churn",
    "diabetes",
    "E-CommereShippingData",
    "Fitness_Club",
    "hazelnut-spread-contaminant-detection",
    "heloc",
    "NATICUSdroid",
    "seismic-bumps",
    "taiwanese_bankruptcy_prediction",
]
BINARY_SMALL_B = [
    "Bank_Customer_Churn",
    "blood-transfusion-service-center",
    "coil2000_insurance_policies",
    "credit-g",
    "in_vehicle_coupon_recommendation",
    "Is-this-a-good-customer",
    "jm1",
    "Marketing_Campaign",
    "online_shoppers_intention",
    "polish_companies_bankruptcy",
    "qsar-biodeg",
]
BINARY_LARGE = [
    "Amazon_employee_access",
    "APSFailure",
    "bank-marketing",
    "credit_card_clients_default",
    "customer_satisfaction_in_airline",
    "Diabetes130US",
    "GiveMeSomeCredit",
    "HR_Analytics_Job_Change_of_Data_Scientists",
    "kddcup09_appetency",
]
assert sorted(BINARY_SMALL_A + BINARY_SMALL_B + BINARY_LARGE) == sorted(BINARY), (
    "the split must cover the binary suite once"
)

#: Multiclass and regression tables whose stored runs were fitted on the Mac and take 1,500-3,300 s
#: there: on a 4-core Kaggle CPU their first bagged fold runs past AutoGluon's per-fold budget
#: (3,600 s / 8), so the bag stops with TimeLimitExceeded (the e5cb05b repro kernel, 2026-11-18).
#: They are reproduced on the Mac with the same worker, commit and pins
#: (``scratchpad repro_mac.sh``; FINDINGS §293), not in a kernel.
MAC_MULTICLASS = ["SDSS17"]
MAC_REGRESSION = [
    "QSAR-TID-11", "miami_housing", "houses", "superconductivity", "Food_Delivery_Time",
    "physiochemical_protein", "diamonds",
]  # fmt: skip
MULTICLASS_KAGGLE = [d for d in MULTICLASS if d not in MAC_MULTICLASS]
REGRESSION_KAGGLE = [d for d in REGRESSION if d not in MAC_REGRESSION]

#: Kernel directory -> {run name: (problem type, datasets)}. Run names are what the merge reads.
KERNELS = {
    "task_lite_repro_pr_bin_small_a": {"fintfm_pr_repro_bin_small_a": ("binary", BINARY_SMALL_A)},
    "task_lite_repro_pr_bin_small_b": {"fintfm_pr_repro_bin_small_b": ("binary", BINARY_SMALL_B)},
    "task_lite_repro_pr_bin_large": {"fintfm_pr_repro_bin_large": ("binary", BINARY_LARGE)},
    "task_lite_repro_pr_mc": {"fintfm_pr_repro_mc": ("multiclass", MULTICLASS_KAGGLE)},
    "task_lite_repro_pr_reg": {"fintfm_pr_repro_reg": ("regression", REGRESSION_KAGGLE)},
}


#: Kernels that run on a Kaggle CPU session: every part of the entry measured on CPU. The
#: multiclass and regression runs were (FINDINGS §255: a T4 repeat changed device and results
#: together), and so were the binary tables of at most 10,000 rows; only the larger binary tables
#: need the T4 they ran on.
CPU_KERNELS = {
    "task_lite_repro_pr_bin_small_a",
    "task_lite_repro_pr_bin_small_b",
    "task_lite_repro_pr_mc",
    "task_lite_repro_pr_reg",
}


def render(kernel: str) -> str:
    """The kernel script text for ``kernel``.

    Args:
        kernel: A key of :data:`KERNELS`.

    Returns:
        The template with the kernel name and run table filled in.
    """
    out = []
    for line in (HERE / "repro_pr_kernel_template.py").read_text().splitlines():
        if "@@RUNS@@" in line:
            body = pprint.pformat(KERNELS[kernel], width=100, sort_dicts=False)
            line = "RUNS: dict[str, tuple[str, list[str]]] = " + body
        line = line.replace("@@KERNEL@@", kernel)
        out.append(line.replace("@@DEVICE@@", "cpu" if kernel in CPU_KERNELS else "gpu"))
    return "\n".join(out) + "\n"


def metadata(kernel: str) -> dict:
    """The ``kernel-metadata.json`` content for ``kernel``."""
    slug = "fintfm-" + kernel.replace("_", "-")
    return {
        "id": f"muhakabartay/{slug}",
        "title": slug,
        "code_file": f"{kernel}.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "false" if kernel in CPU_KERNELS else "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


if __name__ == "__main__":
    for name in KERNELS:
        d = HERE / name
        d.mkdir(exist_ok=True)
        (d / f"{name}.py").write_text(render(name))
        (d / "kernel-metadata.json").write_text(json.dumps(metadata(name), indent=2) + "\n")
        print("wrote", d)
