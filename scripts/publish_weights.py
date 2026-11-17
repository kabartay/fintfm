"""Dry-run publisher for the three checkpoints the TabArena entry uses (openspec 49.6).

Publishing weights is the owner's decision, with its own licence (CLAUDE.md, "Licensing: what
this repo carries"). This script therefore **does nothing by default**: it checks each local
checkpoint against its recorded SHA-256 and prints what an upload would send. Nothing touches
the network unless every one of these holds:

- ``--yes-publish`` is passed,
- ``HF_TOKEN`` is set in the environment (the token is read from there and never stored,
  printed or written anywhere),
- ``--licence`` names an SPDX identifier (the model cards carry ``LICENCE: OWNER DECISION``
  until the owner decides, and the script refuses to upload a card that still says so).

Usage::

    uv run python scripts/publish_weights.py                 # dry run, all three
    uv run python scripts/publish_weights.py --only regression
    uv run python scripts/publish_weights.py --yes-publish --licence <SPDX>   # owner only

Exit codes: 0 success, 1 a checkpoint is missing or its hash differs, 2 refused.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

PLACEHOLDER = "OWNER DECISION"
REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Checkpoint:
    """One checkpoint the TabArena entry loads.

    Attributes:
        key: Short name used by ``--only``.
        repo_id: Hugging Face model repository the checkpoint is (or would be) published to.
        relpath: Path under the runs directory.
        filename: File name inside the Hugging Face repository.
        sha256: Expected SHA-256 of the file (computed 2026-11-15).
        size: Expected size in bytes.
        card: Model card under ``docs/models/``.
        published: True if the file is already public (binary verified 2026-11-15; multiclass and regression published 2026-11-17).
    """

    key: str
    repo_id: str
    relpath: str
    filename: str
    sha256: str
    size: int
    card: str
    published: bool


CHECKPOINTS: tuple[Checkpoint, ...] = (
    Checkpoint(
        key="binary",
        repo_id="kabartay/fintfm-binary",
        relpath="v4-cellattn-labels.pt",
        filename="v4-cellattn-labels.pt",
        sha256="d231b062483c9261aae4c503ad5359db295fc7206fa0183f51763e4953c518d2",
        size=3585704,
        card="docs/models/fintfm-binary.md",
        published=True,
    ),
    Checkpoint(
        key="multiclass",
        repo_id="kabartay/fintfm-multiclass",
        relpath="kaggle-ts-dr74-mc-s0/dr74-mc-s0/dr74-mc-s0.pt",
        filename="dr74-mc-s0.pt",
        sha256="14d5b4d561fbea6e35c7a30dc2644cb8f64f437c25d62da087ee960717cc8f4d",
        size=3598528,
        card="docs/models/fintfm-multiclass.md",
        published=True,
    ),
    Checkpoint(
        key="regression",
        repo_id="kabartay/fintfm-regression",
        relpath="task48-3-kaggle/v4-binned48_3-control.pt",
        filename="v4-binned48_3-control.pt",
        sha256="31d5b3ef49e89bbe405fc85ab1d746a9fd6c3a1c5282e1770111902cc4af526c",
        size=3595783,
        card="docs/models/fintfm-regression.md",
        published=True,
    ),
)


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 of a file, read in 1 MiB blocks.

    Args:
        path: File to hash.

    Returns:
        Lower-case hex digest.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(ckpt: Checkpoint, runs_dir: Path) -> list[str]:
    """Check one checkpoint's presence, size and SHA-256.

    Args:
        ckpt: The checkpoint record.
        runs_dir: Directory the checkpoint's ``relpath`` is resolved against.

    Returns:
        Problems found; empty when the file matches the record.
    """
    path = runs_dir / ckpt.relpath
    if not path.is_file():
        return [f"{ckpt.key}: missing {path}"]
    problems = []
    if path.stat().st_size != ckpt.size:
        problems.append(f"{ckpt.key}: size {path.stat().st_size} != {ckpt.size}")
    digest = sha256_of(path)
    if digest != ckpt.sha256:
        problems.append(f"{ckpt.key}: sha256 {digest} != {ckpt.sha256}")
    return problems


def describe(ckpt: Checkpoint, runs_dir: Path) -> str:
    """Format the line printed for one planned upload.

    Args:
        ckpt: The checkpoint record.
        runs_dir: Directory the checkpoint's ``relpath`` is resolved against.

    Returns:
        Multi-line description.
    """
    state = "already public, would be skipped" if ckpt.published else "not yet public"
    return (
        f"{ckpt.key}: {runs_dir / ckpt.relpath}\n"
        f"    would upload as {ckpt.repo_id}:{ckpt.filename} ({ckpt.size} bytes, {state})\n"
        f"    sha256 {ckpt.sha256}\n"
        f"    card   {ckpt.card} -> README.md"
    )


def _upload(ckpt: Checkpoint, runs_dir: Path, licence: str, token: str) -> None:
    """Create the repository and upload the file and card. Only reached after every guard.

    Args:
        ckpt: The checkpoint record.
        runs_dir: Directory the checkpoint's ``relpath`` is resolved against.
        licence: SPDX identifier the owner chose, written to the card header.
        token: Hugging Face token from the environment.
    """
    from huggingface_hub import HfApi  # imported here so the dry run needs no network stack

    api = HfApi(token=token)
    api.create_repo(ckpt.repo_id, repo_type="model", exist_ok=True)
    card = (REPO_ROOT / ckpt.card).read_text()
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=ckpt.repo_id,
        commit_message=f"Model card ({licence})",
    )
    api.upload_file(
        path_or_fileobj=str(runs_dir / ckpt.relpath),
        path_in_repo=ckpt.filename,
        repo_id=ckpt.repo_id,
        commit_message=f"Add {ckpt.filename} (sha256 {ckpt.sha256[:12]})",
    )


def main(argv: list[str] | None = None, env: dict[str, str] | None = None) -> int:
    """Run the dry run, or the upload when every guard passes.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.
        env: Environment mapping; defaults to ``os.environ``.

    Returns:
        Process exit code (0 success, 1 verification failed, 2 refused).
    """
    env = os.environ if env is None else env
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "runs")
    p.add_argument("--only", choices=[c.key for c in CHECKPOINTS], action="append")
    p.add_argument("--yes-publish", action="store_true", help="actually upload (owner only)")
    p.add_argument("--licence", default=None, help="SPDX identifier chosen by the owner")
    args = p.parse_args(argv)

    chosen = [c for c in CHECKPOINTS if not args.only or c.key in args.only]
    publishing = args.yes_publish

    if publishing:
        token = env.get("HF_TOKEN", "")
        if not token:
            print("refused: --yes-publish needs HF_TOKEN in the environment", file=sys.stderr)
            return 2
        if not args.licence or PLACEHOLDER in args.licence:
            print("refused: --yes-publish needs --licence <SPDX> (owner decision)", file=sys.stderr)
            return 2
        for c in chosen:
            if PLACEHOLDER in (REPO_ROOT / c.card).read_text():
                print(f"refused: {c.card} still says {PLACEHOLDER}", file=sys.stderr)
                return 2

    problems = [m for c in chosen for m in verify(c, args.runs_dir)]
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1

    mode = "PUBLISH" if publishing else "DRY RUN (nothing is uploaded)"
    print(mode)
    for c in chosen:
        print(describe(c, args.runs_dir))
    if not publishing:
        print("all hashes match; pass --yes-publish, --licence and HF_TOKEN to upload")
        return 0

    for c in chosen:
        if c.published:
            print(f"skip {c.key}: already public")
            continue
        _upload(c, args.runs_dir, args.licence, env["HF_TOKEN"])
        print(f"uploaded {c.key}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
