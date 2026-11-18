"""Profile one FinTFM-LinProbe TabArena-Lite fit on an HF Jobs GPU, to see where its time goes.

On HF Jobs ``t4-medium`` the large binary reproduction fits took 1.8 to 3.3x as long as the stored
Kaggle T4 runs, and three tables stopped with ``TimeLimitExceeded`` (FINDINGS §293, 2026-11-18);
``credit_card_clients_default`` did so on ``a10g-large`` too, so a faster GPU did not help. This job
installs exactly what the reproduction kernel installs (it imports the generated kernel at
``FINTFM_REF`` and reuses its ``clone_fork``, ``install``, ``child_env`` and worker), then fits one
table with the registry default config in-process under :mod:`cProfile`, while a thread samples
``nvidia-smi`` utilisation every 5 s. It records the thread settings (``torch``, ``threadpoolctl``,
``os.cpu_count``) beside the profile.

The fit may stop with ``TimeLimitExceeded`` after its first fold; the profile of that fold is the
point, so the stats are written either way. Outputs (``profile.txt``, ``profile.pstats``,
``gpu.csv``, ``threads.json``) are uploaded to the private dataset ``REPO_ID`` under
``profile-<DATASET>-<flavor label>/``.

Launch like ``repro_kernel_on_hf_job.py`` with ``KERNEL``, ``FINTFM_REF``, ``DATASET`` and ``LABEL``
in the environment and ``HF_TOKEN`` as a secret.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import types
import urllib.request
from pathlib import Path

#: Private dataset repository that receives the outputs.
REPO_ID = "kabartay/fintfm-lite-repro"
OUT = Path("/work/profile")

FIT = """\
import cProfile, json, os, pstats, sys, time
out = sys.argv[2]
info = {"cpu_count": os.cpu_count(), "env": {k: os.environ.get(k) for k in
        ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
import torch
info["torch_threads"] = torch.get_num_threads()
info["torch_interop_threads"] = torch.get_num_interop_threads()
info["cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
try:
    from threadpoolctl import threadpool_info
    info["threadpools"] = threadpool_info()
except Exception as e:  # recorded, not fatal
    info["threadpools"] = repr(e)
open(f"{out}/threads.json", "w").write(json.dumps(info, indent=1, default=str))
print("PROFILE threads", json.dumps({k: v for k, v in info.items() if k != "threadpools"}), flush=True)

from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.contexts import TabArenaContext

prof = cProfile.Profile()
t0 = time.time()
prof.enable()
try:
    TabArenaContext().build_and_run_jobs(
        TabArenaV0pt1ExperimentBundle(models=[("FinTFM-LinProbe", 0)]).build_experiments(),
        expname="/tmp/profile_results",
        subset="lite",
        build_kwargs={"dataset_names": [sys.argv[1]]},
        new_result_prefix="[New] ",
        debug_mode=True,
    )
    status = "completed"
except BaseException as e:  # TimeLimitExceeded is expected and still profiled
    status = f"stopped: {type(e).__name__}"
finally:
    prof.disable()
    prof.dump_stats(f"{out}/profile.pstats")
    with open(f"{out}/profile.txt", "w") as f:
        st = pstats.Stats(prof, stream=f).sort_stats("cumulative")
        f.write(f"{sys.argv[1]}: {status} after {time.time() - t0:.0f} s\\n\\n")
        st.print_stats(80)
        st.sort_stats("tottime").print_stats(60)
    print(f"PROFILE {sys.argv[1]}: {status} after {time.time() - t0:.0f} s", flush=True)
"""


def load_kernel(kernel: str, ref: str) -> types.ModuleType:
    """Import the generated kernel script at ``ref`` as a module, without running its main.

    Args:
        kernel: Kernel directory name under ``scripts/kaggle/``.
        ref: fintfm git commit.

    Returns:
        The kernel module.
    """
    url = f"https://raw.githubusercontent.com/kabartay/fintfm/{ref}/scripts/kaggle/{kernel}/{kernel}.py"
    mod = types.ModuleType("kernel")
    mod.__dict__["__name__"] = "kernel"
    # The generated kernel at a pinned commit of this repository, run as a module: trusted source.
    code = compile(urllib.request.urlopen(url, timeout=60).read().decode(), url, "exec")
    exec(code, mod.__dict__)  # noqa: S102
    return mod


def sample_gpu(path: Path, stop: threading.Event) -> None:
    """Append ``nvidia-smi`` utilisation and memory to ``path`` every 5 s until ``stop`` is set.

    Args:
        path: CSV file to append to.
        stop: Event that ends the sampling.
    """
    query = "timestamp,utilization.gpu,utilization.memory,memory.used"
    with path.open("w") as f:
        f.write(query + "\n")
        while not stop.is_set():
            r = subprocess.run(
                ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=False,
            )
            f.write(r.stdout)
            f.flush()
            stop.wait(5)


def upload(folder: str) -> None:
    """Upload ``OUT`` to ``REPO_ID`` under ``folder``.

    Args:
        folder: Folder name in the repository.
    """
    from huggingface_hub import HfApi

    api = HfApi()  # reads HF_TOKEN from the environment
    api.upload_folder(
        folder_path=str(OUT),
        repo_id=REPO_ID,
        repo_type="dataset",
        path_in_repo=folder,
        commit_message=f"{folder}: profile of one fit",
    )
    print(f"uploaded {OUT} to datasets/{REPO_ID}/{folder}", flush=True)


def gpu_summary(path: Path) -> str:
    """Mean and share-of-samples-above-50% GPU utilisation from the sampled CSV.

    Args:
        path: CSV written by :func:`sample_gpu`.

    Returns:
        A one-line summary.
    """
    util = []
    for line in path.read_text().splitlines()[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4 and parts[1].isdigit():
            util.append(int(parts[1]))
    if not util:
        return "no GPU samples"
    busy = sum(u > 50 for u in util) / len(util)
    return (
        f"{len(util)} samples, mean GPU util {sum(util) / len(util):.1f}%, above 50% in {busy:.0%}"
    )


def main() -> int:
    """Install as the kernel does, profile one fit, upload; return 0."""
    dataset, label = os.environ["DATASET"], os.environ["LABEL"]
    k = load_kernel(os.environ["KERNEL"], os.environ["FINTFM_REF"])
    fork = Path("/tmp/tabarena")
    k.clone_fork(fork)
    k.install(fork)
    OUT.mkdir(parents=True, exist_ok=True)
    script = Path("/tmp/profile_fit_worker.py")
    script.write_text(FIT)
    env, removed = k.child_env()
    print(f"PROFILE removed FINTFM_* variables: {removed}", flush=True)
    stop = threading.Event()
    sampler = threading.Thread(target=sample_gpu, args=(OUT / "gpu.csv", stop), daemon=True)
    sampler.start()
    t0 = time.time()
    try:
        code = subprocess.run(
            [sys.executable, "-P", str(script), dataset, str(OUT)],
            cwd=fork,
            env=env,
            check=False,
        ).returncode
        print(f"PROFILE worker exit {code}, {time.time() - t0:.0f} s wall", flush=True)
    finally:
        stop.set()
        sampler.join(timeout=10)
        summary = gpu_summary(OUT / "gpu.csv")
        print(f"PROFILE gpu: {summary}", flush=True)
        (OUT / "gpu_summary.json").write_text(json.dumps({"summary": summary}))
        upload(f"profile-{dataset}-{label}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
