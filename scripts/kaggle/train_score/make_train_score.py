"""Write one Kaggle kernel directory per train-and-score arm from train_score.py."""

import json
from pathlib import Path

COMMIT = "d05ea19"
#: E1 (FINDINGS §171): fp16, split attention, DDP on a T4x2 at 4 tasks per rank (global batch 8)
E1 = ["--amp", "fp16", "--attn-impl", "split", "--loader", "stream", "--ddp", "--batch-size", "4"]
ARMS = {
    "e1-v48-s0": (E1 + ["--seed", "0", "--steps", "6000"], True),
    "e1-v48-s1": (E1 + ["--seed", "1", "--steps", "6000"], True),
    "e1-v480-s0": (E1 + ["--seed", "0", "--steps", "60000"], True),
    "e1-v480-s1": (E1 + ["--seed", "1", "--steps", "60000"], True),
}
OLD_ARMS = {
    # DR4.4 replicates of the published recipe (seed 0 also re-checks reproducibility)
    "pub-s0": ["--seed", "0"],
    "pub-s1": ["--seed", "1"],
    "pub-s2": ["--seed", "2"],
    # AU2.8: the published recipe without the cell-attention block, same prior and protocol
    "nocell-s0": ["--seed", "0", "--n-cell-blocks", "0", "--no-cell-labels"],
}
HERE = Path(__file__).parent
TEMPLATE = (HERE / "train_score.py").read_text()

for arm, (flags, ddp) in ARMS.items():
    d = HERE.parent / f"ts_{arm.replace('-', '_')}"
    d.mkdir(exist_ok=True)
    code = (TEMPLATE.replace("__COMMIT__", COMMIT).replace("__ARM__", arm)
            .replace("[]  # __FLAGS__", json.dumps(flags))
            .replace("False  # __DDP__", str(ddp)).replace("False  # __PART__", "False"))
    (d / "kernel.py").write_text(code)
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"muhakabartay/fintfm-ts-{arm}", "title": f"fintfm-ts-{arm}",
        "code_file": "kernel.py", "language": "python", "kernel_type": "script",
        "is_private": "true", "enable_gpu": "true", "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": ["sebastiantomczak10/v4-group-corporate-bankruptcy"],
        "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }, indent=2) + "\n")
    print(d)
