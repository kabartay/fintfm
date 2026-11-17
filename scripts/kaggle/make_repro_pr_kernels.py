"""Write the two TabArena-Lite reproduction kernels from ``repro_pr_kernel_template.py``.

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

#: The binary suite in two kernels, balanced by the stored fit times of the §287 entry's binary runs
#: (about 5.1 h each, 2026-11-17): one kernel needed about 10.3 h against the 11 h budget.
BINARY_A = [
    "Amazon_employee_access",
    "APSFailure",
    "bank-marketing",
    "churn",
    "credit_card_clients_default",
    "customer_satisfaction_in_airline",
    "diabetes",
    "E-CommereShippingData",
    "Fitness_Club",
    "heloc",
    "HR_Analytics_Job_Change_of_Data_Scientists",
    "in_vehicle_coupon_recommendation",
    "kddcup09_appetency",
    "polish_companies_bankruptcy",
    "qsar-biodeg",
]
BINARY_B = [
    "Bank_Customer_Churn",
    "Bioresponse",
    "blood-transfusion-service-center",
    "coil2000_insurance_policies",
    "credit-g",
    "Diabetes130US",
    "GiveMeSomeCredit",
    "hazelnut-spread-contaminant-detection",
    "Is-this-a-good-customer",
    "jm1",
    "Marketing_Campaign",
    "NATICUSdroid",
    "online_shoppers_intention",
    "seismic-bumps",
    "taiwanese_bankruptcy_prediction",
]
assert sorted(BINARY_A + BINARY_B) == sorted(BINARY), "the split must cover the binary suite once"

#: Kernel directory -> {run name: (problem type, datasets)}. Run names are what the merge reads.
KERNELS = {
    "task_lite_repro_pr_bin_a": {"fintfm_pr_repro_bin_a": ("binary", BINARY_A)},
    "task_lite_repro_pr_bin_b": {"fintfm_pr_repro_bin_b": ("binary", BINARY_B)},
    "task_lite_repro_pr_mcreg": {
        "fintfm_pr_repro_mc": ("multiclass", MULTICLASS),
        "fintfm_pr_repro_reg": ("regression", REGRESSION),
    },
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
        out.append(line.replace("@@KERNEL@@", kernel))
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
        "enable_gpu": "true",
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
