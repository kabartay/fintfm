"""Run one generated TabArena-Lite kernel on a RunPod GPU pod, copy its outputs back, delete the pod.

The RunPod counterpart of ``scripts/hf/repro_kernel_on_hf_job.py``, and it runs that same wrapper
inside the pod with ``UPLOAD=0``, so the kernel, its pinning and its deadline handling are
identical to an HF Jobs run. Only the transport differs:

1. create a pod (default: one ``NVIDIA A40``, Secure Cloud, host CUDA >= 13.2 for the cu132 torch
   wheels) whose container fetches the wrapper at ``--ref`` and runs it, writing ``/work/EXIT``
   when it ends;
2. poll the pod through the REST API's ``exec`` endpoint (no SSH, no public port, no secret in the
   pod: the HF token is not needed because nothing is uploaded from the pod);
3. tar ``/work`` inside the pod and copy it back in base64 chunks (each ``exec`` reply is capped at
   1 MiB), checking the SHA-256;
4. delete the pod, always: on success, on failure, on Ctrl-C, and when ``--max-hours`` passes.

The API key is read from ``RUNPOD_API_KEY`` (``set -a; . ./.env; set +a`` in the calling command)
and is never printed or sent to the pod. It needs only Pods read/write, Catalog read and Billing
read.

Example (the credit_card table of the BLAS-limit kernel, to compare with its HF result)::

    set -a; . ./.env; set +a; python3 scripts/runpod/run_kernel_on_pod.py \\
        --name fintfm-rp-threads-cc --kernel task_lite_threads_bin --ref <fintfm commit> \\
        --datasets credit_card_clients_default --out runs/runpod/threads-cc

Exit status: the kernel's (``124`` at its deadline), ``3`` when the pod never produced an exit
code, ``2`` on a usage error.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shlex
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

#: RunPod REST API root (the older ``rest.runpod.io`` host no longer resolves).
API = "https://api.runpod.io/v2"
#: Wrapper that fetches and runs the kernel; the same file HF Jobs runs.
WRAPPER = "scripts/hf/repro_kernel_on_hf_job.py"
#: Bytes per copied chunk; base64 makes it 4/3 larger, which stays under the 1 MiB exec cap.
CHUNK = 512 * 1024
#: Seconds between status polls.
POLL_S = 60


class RunPod:
    """Minimal client for the RunPod v2 REST API.

    Args:
        key: API key. Kept only in the ``Authorization`` header, never logged.
    """

    def __init__(self, key: str) -> None:
        self._headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def call(self, method: str, path: str, body: dict | None = None, timeout: int = 60) -> dict:
        """Send one request and return the decoded JSON reply (``{}`` for an empty body).

        Args:
            method: HTTP method.
            path: Path under :data:`API`, starting with ``/``.
            body: JSON body, if any.
            timeout: Socket timeout in seconds.

        Returns:
            The decoded reply.

        Raises:
            RuntimeError: The API answered with an error status; the message carries the status and
                the API's reply text (which never contains the key).
        """
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(API + path, data=data, method=method, headers=self._headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                text = resp.read().decode()
        except urllib.error.HTTPError as err:
            raise RuntimeError(
                f"{method} {path}: HTTP {err.code}: {err.read().decode()[:500]}"
            ) from None
        return json.loads(text) if text.strip() else {}

    def exec(self, pod_id: str, script: str, timeout_s: int = 25) -> dict:
        """Run ``sh -c script`` in the pod's container.

        Args:
            pod_id: Pod ID.
            script: Shell script.
            timeout_s: How long the API waits for it (1-25 s).

        Returns:
            The API's reply: ``exitCode``, ``stdout``, ``stderr``, ``timedOut``, ``truncated``.
        """
        body = {"command": ["sh", "-c", script], "timeoutSeconds": timeout_s}
        return self.call("POST", f"/pods/{pod_id}/exec", body, timeout=timeout_s + 30)


def container_command(ref: str) -> str:
    """The pod's entrypoint script: run the wrapper, record its exit code, then idle.

    The container idles after the run instead of exiting, so that the outputs survive until they
    are copied back; the launcher deletes the pod afterwards.

    Args:
        ref: fintfm commit to fetch the wrapper from.

    Returns:
        A ``bash -c`` script.
    """
    url = f"https://raw.githubusercontent.com/kabartay/fintfm/{ref}/{WRAPPER}"
    return (
        "mkdir -p /work; "
        f"(curl -sfL {shlex.quote(url)} -o /tmp/wrapper.py && python /tmp/wrapper.py) > /work/pod.log 2>&1; "
        "echo $? > /work/EXIT; sleep infinity"
    )


def create_pod(api: RunPod, args: argparse.Namespace) -> dict:
    """Create the pod.

    Args:
        api: Client.
        args: Parsed command line.

    Returns:
        The created pod as the API describes it.
    """
    env = {
        "KERNEL": args.kernel,
        "FINTFM_REF": args.ref,
        "UPLOAD": "0",
        "DEADLINE_S": str(args.deadline_s),
    }
    if args.datasets:
        env["DATASETS"] = args.datasets
    body = {
        "name": args.name,
        "image": args.image,
        "entrypoint": ["bash", "-c", container_command(args.ref)],
        "env": env,
        "disk": args.disk_gb,
        "cloud": args.cloud,
        "gpu": {
            "id": args.gpu,
            "count": 1,
            "minCudaVersion": args.min_cuda,
            "minVcpuCountPerGpu": args.min_vcpu,
            "minRamPerGpu": args.min_ram_gb,
        },
    }
    return api.call("POST", "/pods", body)


def fetch_outputs(api: RunPod, pod_id: str, out: Path) -> None:
    """Copy the pod's ``/work`` to ``out`` and unpack it there.

    Args:
        api: Client.
        pod_id: Pod ID.
        out: Local directory; created if missing.

    Raises:
        RuntimeError: The archive could not be made, or its checksum does not match.
    """
    made = api.exec(
        pod_id,
        "cd /work && tar czf /tmp/out.tgz --exclude=repro_worker.py . && "
        "wc -c < /tmp/out.tgz && sha256sum /tmp/out.tgz | cut -d' ' -f1",
    )
    if made.get("exitCode") != 0:
        raise RuntimeError(f"tar in the pod failed: {made.get('stderr', '')[:500]}")
    size_text, digest = made["stdout"].split()
    size = int(size_text)
    blob = bytearray()
    for i in range((size + CHUNK - 1) // CHUNK):
        part = api.exec(
            pod_id, f"dd if=/tmp/out.tgz bs={CHUNK} skip={i} count=1 2>/dev/null | base64 -w0"
        )
        if part.get("exitCode") != 0 or part.get("truncated"):
            raise RuntimeError(f"chunk {i} failed: {part.get('stderr', '')[:300]}")
        blob += base64.b64decode(part["stdout"])
    if len(blob) != size or hashlib.sha256(blob).hexdigest() != digest:
        raise RuntimeError(
            f"archive mismatch: got {len(blob)} of {size} bytes or a different SHA-256"
        )
    out.mkdir(parents=True, exist_ok=True)
    archive = out / "pod_outputs.tgz"
    archive.write_bytes(blob)
    with tarfile.open(archive) as tar:
        tar.extractall(out, filter="data")
    print(f"copied {size} bytes to {out}", flush=True)


def log(msg: str) -> None:
    """Print one timestamped progress line."""
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def run(api: RunPod, args: argparse.Namespace) -> int:
    """Create, watch, copy back and delete; return the kernel's exit status (``3`` if none).

    Args:
        api: Client.
        args: Parsed command line.

    Returns:
        Exit status.
    """
    pod = create_pod(api, args)
    pod_id = pod["id"]
    log(f"created pod {pod_id} ({args.gpu}, {args.cloud}, ${pod.get('cost', '?')}/h)")
    started = time.monotonic()
    code: int | None = None
    last_line = ""
    try:
        while time.monotonic() - started < args.max_hours * 3600:
            time.sleep(POLL_S)
            info = api.call("GET", f"/pods/{pod_id}")
            status = info.get("status")
            if status in ("ERROR", "TERMINATED", "EXITED"):
                log(f"pod status {status}; giving up")
                break
            if status != "RUNNING":
                log(f"pod status {status}")
                continue
            try:
                probe = api.exec(
                    pod_id,
                    "cat /work/EXIT 2>/dev/null; echo ---; tail -n 1 /work/pod.log 2>/dev/null",
                )
            except RuntimeError as err:  # container still starting (409) or exec not ready (503)
                log(f"exec not ready: {str(err)[:120]}")
                continue
            head, _, tail = probe.get("stdout", "").partition("---")
            line = tail.strip()
            if line and line != last_line:
                log(f"pod: {line[:200]}")
                last_line = line
            if head.strip():
                code = int(head.strip())
                log(f"kernel exit {code} after {(time.monotonic() - started) / 60:.0f} min")
                break
        else:
            log(f"--max-hours {args.max_hours} passed; deleting the pod")
        if code is not None or args.fetch_on_failure:
            try:
                fetch_outputs(api, pod_id, Path(args.out))
            except RuntimeError as err:
                log(f"copy failed: {err}")
    finally:
        hours = (time.monotonic() - started) / 3600
        try:
            api.call("DELETE", f"/pods/{pod_id}")
            log(f"deleted pod {pod_id}; about {hours:.2f} h at ${pod.get('cost', '?')}/h")
        except RuntimeError as err:
            log(f"DELETE FAILED, delete pod {pod_id} in the console: {err}")
    return 3 if code is None else code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--name", required=True, help="pod name, e.g. fintfm-rp-threads-cc")
    p.add_argument("--kernel", required=True, help="kernel directory under scripts/kaggle/")
    p.add_argument("--ref", required=True, help="fintfm commit holding the kernel and the wrapper")
    p.add_argument("--datasets", default="", help="comma-separated subset of the kernel's datasets")
    p.add_argument("--out", required=True, help="local directory for the outputs")
    p.add_argument("--gpu", default="NVIDIA A40", help="RunPod GPU type ID")
    p.add_argument("--cloud", default="SECURE", choices=["SECURE", "COMMUNITY"])
    p.add_argument("--image", default="python:3.13", help="container image (the HF jobs' image)")
    p.add_argument(
        "--min-cuda", default="13.2", help="lowest host CUDA version (torch cu132 wheels)"
    )
    p.add_argument("--min-vcpu", type=int, default=8, help="lowest vCPU count on the host per GPU")
    p.add_argument("--min-ram-gb", type=int, default=32, help="lowest host RAM per GPU in GB")
    p.add_argument("--disk-gb", type=int, default=50, help="container disk in GB")
    p.add_argument(
        "--deadline-s", type=int, default=5 * 3600, help="the wrapper's deadline for the kernel"
    )
    p.add_argument(
        "--max-hours",
        type=float,
        default=6.0,
        help="delete the pod after this long, whatever happens",
    )
    p.add_argument(
        "--fetch-on-failure",
        action="store_true",
        help="copy outputs back even without an exit code",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        print("RUNPOD_API_KEY is not set; source .env inside the command", file=sys.stderr)
        return 2
    if args.max_hours * 3600 < args.deadline_s + 900:
        print("--max-hours must leave 15 min after --deadline-s for the copy", file=sys.stderr)
        return 2
    return run(RunPod(key), args)


if __name__ == "__main__":
    sys.exit(main())
