"""Run one generated TabArena-Lite reproduction kernel on Hugging Face Jobs, then upload its outputs.

The reproduction kernels (``scripts/kaggle/task_lite_repro_pr_*``) are written for Kaggle and write
under ``/kaggle/working``. When Kaggle's weekly GPU quota is exhausted, the T4 kernel runs here
instead, on an HF Jobs ``t4-medium`` (same GPU model, 30 GB of RAM like a Kaggle T4 session). This
wrapper changes nothing in the kernel except its working directory:

1. download the kernel script from GitHub at ``FINTFM_REF`` (a commit, so the code is pinned);
2. point ``WORK`` at ``/work`` (the kernel also clones the fork into a temporary directory when
   ``/kaggle`` does not exist);
3. run it with a deadline shorter than the job's timeout, so step 4 always runs;
4. upload every file in ``/work`` except the worker script to the private dataset repo
   ``REPO_ID`` under ``<kernel>/``, whether the kernel succeeded or not.

Launch (the token is passed as a job secret from the caller's environment, never printed)::

    hf jobs run --detach --flavor t4-medium --timeout 6h --secrets HF_TOKEN \\
        -e KERNEL=task_lite_repro_pr_bin_large -e FINTFM_REF=<commit> \\
        python:3.13 bash -c "pip install -q huggingface_hub && \\
        curl -sL https://raw.githubusercontent.com/kabartay/fintfm/<commit>/scripts/hf/repro_kernel_on_hf_job.py \\
        | python -"

Exit status: the kernel's, or 124 when it was stopped at the deadline.
"""

from __future__ import annotations

import os
import subprocess
import sys
import urllib.request
from pathlib import Path

#: Private dataset repository that receives the outputs.
REPO_ID = "kabartay/fintfm-lite-repro"
WORK = Path("/work")
#: Seconds the kernel may run; the job's own timeout (6 h) leaves room for the upload.
DEADLINE_S = int(os.environ.get("DEADLINE_S", str(5 * 3600 + 1800)))


def fetch_kernel(kernel: str, ref: str) -> Path:
    """Download the generated kernel script and repoint its working directory.

    Args:
        kernel: Kernel directory name under ``scripts/kaggle/``.
        ref: fintfm git commit to read it from.

    Returns:
        Path of the patched script.

    Raises:
        SystemExit: The script does not contain the expected ``WORK`` line, so the patch would
            silently not apply.
    """
    url = f"https://raw.githubusercontent.com/kabartay/fintfm/{ref}/scripts/kaggle/{kernel}/{kernel}.py"
    text = urllib.request.urlopen(url, timeout=60).read().decode()
    old = 'WORK = Path("/kaggle/working")'
    if text.count(old) != 1:
        sys.exit(f"{url}: expected exactly one {old!r}")
    path = Path("/tmp") / f"{kernel}.py"
    path.write_text(text.replace(old, f'WORK = Path("{WORK}")'))
    return path


def upload(kernel: str) -> None:
    """Upload ``/work``'s outputs to ``REPO_ID`` under ``<kernel>/``.

    Args:
        kernel: Kernel directory name, used as the folder in the repository.
    """
    from huggingface_hub import HfApi

    api = HfApi()  # reads HF_TOKEN from the environment
    api.create_repo(REPO_ID, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(
        folder_path=str(WORK),
        repo_id=REPO_ID,
        repo_type="dataset",
        path_in_repo=kernel,
        ignore_patterns=["repro_worker.py"],
        commit_message=f"{kernel}: outputs of the HF Jobs run",
    )
    print(f"uploaded {WORK} to datasets/{REPO_ID}/{kernel}", flush=True)


def main() -> int:
    """Fetch, run with a deadline, upload; return the kernel's exit status."""
    kernel, ref = os.environ["KERNEL"], os.environ["FINTFM_REF"]
    WORK.mkdir(parents=True, exist_ok=True)
    script = fetch_kernel(kernel, ref)
    print(f"running {kernel} from fintfm {ref}, deadline {DEADLINE_S} s", flush=True)
    try:
        code = subprocess.run(
            [sys.executable, str(script)], timeout=DEADLINE_S, check=False
        ).returncode
    except subprocess.TimeoutExpired:
        print("deadline reached; uploading what exists", flush=True)
        code = 124
    finally:
        upload(kernel)
    return code


if __name__ == "__main__":
    sys.exit(main())
