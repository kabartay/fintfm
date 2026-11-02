"""Kaggle kernel: DR3.11's fine-tune-per-leaf arm, §169's partition with §163's fine-tuning, on GPU.

§169 partitions the training rows with a depth-2 tree (5% leaf floor) and draws one context per
leaf; this arm adds §163's 200-step fine-tuning inside every leaf. Folds 0 and 1 only: the
quota left after 2026-11-01's runs (about 4.7 h) does not cover five folds, so folds 2-4 run at
the next reset. Pinned to 027e0cc. The checkpoint is the public kabartay/fintfm-binary
file, verified by SHA-256 against runs/v4-cellattn-labels.pt; the data is V4FinBench's own public
Kaggle dataset. Outputs (per-fold predictions, v4_protocol.json) land in /kaggle/working/FTLEAF/.
"""

import hashlib
import subprocess
import sys
import urllib.request
from pathlib import Path

CKPT_URL = "https://huggingface.co/kabartay/fintfm-binary/resolve/f116bfd43a2b15c65ed3551ea8c38e3364629ddc/v4-cellattn-labels.pt"
CKPT_SHA256 = "d231b062483c9261aae4c503ad5359db295fc7206fa0183f51763e4953c518d2"
CONFIG = """large_n:
  method: partition
inference:
  finetune_steps: 200
  finetune_lr: 1.0e-5
  finetune_batch: 2
  finetune_query: 512
  query_chunk: 512        # identity-preserving (tests); a 2,048 chunk did not fit a T4 after fine-tuning
  feature_chunk: 8        # identity-preserving (tests)
"""


def main():
    hits = sorted(Path("/kaggle/input").rglob("company_years_h1.parquet"))
    if len(hits) != 1:
        sys.exit(f"expected one company_years_h1.parquet under /kaggle/input, found {hits}")
    root = hits[0].parent
    print(f"data: {root}", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                    "fintfm @ git+https://github.com/kabartay/fintfm.git@027e0cc"], check=True)
    import torch
    print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}", flush=True)
    if not torch.cuda.is_available():
        sys.exit("no GPU in this session (quota exhausted?) -- refusing a CPU run")
    ckpt = Path("/kaggle/working/v4-cellattn-labels.pt")
    urllib.request.urlretrieve(CKPT_URL, ckpt)
    digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    if digest != CKPT_SHA256:
        sys.exit(f"checkpoint hash {digest} != {CKPT_SHA256}")
    cfg_path = Path("/kaggle/working/FTLEAF.yaml")
    cfg_path.write_text(CONFIG)
    from fintfm.config import load_config
    from fintfm.experiments.v4_protocol import run, summarise
    record = run(str(ckpt), Path("/kaggle/working/FTLEAF"), horizon=0, folds=(0, 1),
                 root=root, with_boosting=False, classical=("logistic_regression",),
                 cfg=load_config(cfg_path), device="cuda", drop_columns=())
    print(summarise(record), flush=True)
    ckpt.unlink()  # the public file; no need to ship it back
    print("done")


if __name__ == "__main__":
    main()
