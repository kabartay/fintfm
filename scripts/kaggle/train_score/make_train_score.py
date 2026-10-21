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
    # DR6.2 / DR6.3 screening (FINDINGS §175): E1's 48k recipe at seed 0, one lever each,
    # against E1's 48k arms as controls
    "dr62-induced-s0": (E1 + ["--seed", "0", "--steps", "6000", "--cell-row-attn", "induced"],
                        True),
    "dr63-cls-s0": (E1 + ["--seed", "0", "--steps", "6000", "--readout", "cls", "--head-mlp"],
                    True),
}
#: Arms pinned at another commit: 6916a89 adds the DR6.2/DR6.3 flags on top of d05ea19 and
#: nothing else under src/fintfm/modeling or src/fintfm/prior, with defaults unchanged
COMMITS = {"dr62-induced-s0": "6916a89", "dr63-cls-s0": "6916a89"}
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
    code = (TEMPLATE.replace("__COMMIT__", COMMITS.get(arm, COMMIT)).replace("__ARM__", arm)
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
