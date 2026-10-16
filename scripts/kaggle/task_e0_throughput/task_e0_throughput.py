"""Kaggle kernel: design review DR5.1 (E0), training throughput on a Kaggle GPU.

Measures tasks per GPU-hour at the published checkpoint's recipe
(configs/published/v4-cellattn-labels.yaml) for each setting E0 must decide: precision
(fp32, fp16 with loss scaling), row attention (masked, split), rows mixed {256, 512, 1024}
against pinned at 1,024, padding to the batch's widest task, and the streaming loader's
CPU workers. Each arm is `fintfm-train --benchmark-steps 50` (3 untimed warm-up steps), which
reports tasks/s, the CPU prior-sampling share and peak memory as one JSON line. Before timing,
it checks the settings agree on one fixed batch, because E0 adopts a setting only if outputs
agree and tasks per GPU-hour rise by 1.3x or more. If the session has two GPUs it also runs a
20-step DDP smoke test, the NCCL path that has only run on CPU gloo so far. Nothing is
trained to keep; no data is read. Results land in /kaggle/working/e0/.
"""

import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

COMMIT = "b2b836e"
RECIPE_URL = (f"https://raw.githubusercontent.com/kabartay/fintfm/{COMMIT}/"
              "configs/published/v4-cellattn-labels.yaml")
OUT = Path("/kaggle/working/e0")

ARMS = [
    ("fp32_masked_mixed", []),
    ("fp16_masked_mixed", ["--amp", "fp16"]),
    ("fp32_split_mixed", ["--attn-impl", "split"]),
    ("fp16_split_mixed", ["--amp", "fp16", "--attn-impl", "split"]),
    ("fp32_masked_1024", ["--n-rows-choices", "1024"]),
    ("fp16_split_1024", ["--amp", "fp16", "--attn-impl", "split", "--n-rows-choices", "1024"]),
    ("fp16_split_mixed_batchwidth", ["--amp", "fp16", "--attn-impl", "split",
                                     "--pad-to-batch-width"]),
    ("fp16_split_mixed_stream_w0", ["--amp", "fp16", "--attn-impl", "split",
                                    "--loader", "stream", "--loader-workers", "0"]),
    ("fp16_split_mixed_stream_w2", ["--amp", "fp16", "--attn-impl", "split",
                                    "--loader", "stream", "--loader-workers", "2"]),
]


def sh(cmd: list[str], log: Path) -> int:
    """Run a command, streaming its output to the kernel log and a file."""
    print("$ " + " ".join(cmd), flush=True)
    with log.open("w") as fh:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False
        )
        fh.write(proc.stdout)
    print(proc.stdout[-4000:], flush=True)
    return proc.returncode


def agreement(recipe: Path) -> dict:
    """Max |logit| difference of each setting against fp32 masked, on one fixed batch."""
    import numpy as np
    import torch

    from fintfm.config import load_config
    from fintfm.modeling.model import FinancialTFM
    from fintfm.modeling.train import build_configs, build_parser
    from fintfm.prior.mixture import sample_batch

    args = build_parser().parse_args(["--config", str(recipe), "--out", "/tmp/x.pt"])
    model_cfg, prior_cfg, _ = build_configs(args, load_config(recipe))
    batch = sample_batch(np.random.default_rng(123), prior_cfg, 4)
    X = torch.as_tensor(batch.X).cuda()
    y = torch.as_tensor(batch.y).cuda()
    nc = torch.as_tensor(batch.n_classes).cuda()
    out = {}
    ref = None
    for impl in ("masked", "split"):
        torch.manual_seed(0)
        cfg = type(model_cfg)(**{**model_cfg.__dict__, "attn_impl": impl})
        model = FinancialTFM(cfg).cuda().eval()
        model.feature_chunk = 4
        for amp in ("fp32", "fp16"):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp == "fp16"):
                logits = model(X, y, batch.n_ctx, nc, column_id_seed=0).float()
            q = logits[:, batch.n_ctx:]
            q = torch.where(torch.isfinite(q), q, torch.zeros_like(q))
            if ref is None:
                ref = q
            out[f"{amp}_{impl}"] = float((q - ref).abs().max())
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # ask nvidia-smi, not torch: initialising CUDA here would hold GPU 0 memory for the run
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True, check=False)
    names = [n.strip() for n in smi.stdout.splitlines() if n.strip()]
    if smi.returncode != 0 or not names:
        sys.exit("no GPU in this session (quota exhausted?) -- refusing a CPU run")
    print(f"GPUs: {names}; CPUs: {os.cpu_count()}", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                    f"fintfm @ git+https://github.com/kabartay/fintfm.git@{COMMIT}"], check=True)
    recipe = OUT / "v4-cellattn-labels.yaml"
    urllib.request.urlretrieve(RECIPE_URL, recipe)

    summary = {"commit": COMMIT, "gpus": names, "cpus": os.cpu_count()}
    # v1 ran the agreement check in this process first; its 1.3 GB of CUDA memory then sat on
    # GPU 0 through every timed arm and they all ran out of memory. Nothing here touches CUDA
    # until the arms are done, and the check runs last in its own process.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    base = [sys.executable, "-m", "fintfm.modeling.train", "--config", str(recipe),
            "--device", "cuda", "--benchmark-steps", "50", "--out", str(OUT / "unused.pt")]
    results = {}
    for name, extra in ARMS:
        log = OUT / f"{name}.log"
        rc = sh(base + extra, log)
        chunk = 8
        if rc != 0 and "OutOfMemoryError" in log.read_text():
            # the recipe at batch 8 only just fits a 14.56 GB Kaggle T4 (v1 ran out at
            # 14.2 GB on the GPU no other process touched); feature_chunk 4 is
            # identity-preserving and lowers the within-feature attention peak
            chunk = 4
            rc = sh(base + extra + ["--feature-chunk", "4"], log)
        lines = [ln for ln in log.read_text().splitlines() if ln.startswith("{")]
        results[name] = {"exit": rc, "feature_chunk": chunk,
                         "result": json.loads(lines[-1]) if lines else None}
    summary["arms"] = results

    if len(names) >= 2:
        ddp = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node",
               "2", "-m", "fintfm.modeling.train", "--config", str(recipe), "--device", "cuda",
               "--amp", "fp16", "--attn-impl", "split", "--loader", "stream",
               "--loader-workers", "1", "--ddp", "--feature-chunk", "4",
               "--steps", "20", "--log-every", "5",
               "--eval-every", "1000", "--checkpoint-every", "10",
               "--out", str(OUT / "ddp_smoke.pt")]
        summary["ddp_smoke_exit"] = sh(ddp, OUT / "ddp_smoke.log")

    check = OUT / "agreement.json"
    rc = sh([sys.executable, __file__, "--agreement", str(recipe), str(check)], OUT / "agreement.log")
    summary["agreement_exit"] = rc
    if check.exists():
        summary["agreement_max_abs_logit_diff"] = json.loads(check.read_text())
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    for p in OUT.glob("*.pt*"):
        p.unlink()
    print("done")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--agreement":
        Path(sys.argv[3]).write_text(json.dumps(agreement(Path(sys.argv[2]))))
    else:
        main()
