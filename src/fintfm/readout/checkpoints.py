"""The three published checkpoints of the linear readout, and how to fetch one.

A checkpoint reference is a plain dict, so it survives YAML and pickle round trips in a config. A
published one names a Hugging Face file by repo, filename, commit and SHA-256. ``huggingface_hub`` is not
a core dependency of fintfm; it is imported only when a reference is resolved.
"""

from __future__ import annotations

import hashlib
from typing import Any

#: A checkpoint reference: a Hugging Face file named by repo, filename, commit and SHA-256.
CheckpointRef = dict[str, str]

_PUBLISHED_KEYS = frozenset({"repo_id", "filename", "revision", "sha256"})

#: The published binary checkpoint, Apache-2.0 and ungated, pinned to a **commit** rather than
#: a branch. A moving reference would let the weights change underneath a recorded leaderboard
#: number, which is the one thing a benchmark entry cannot tolerate.
BINARY_CHECKPOINT: CheckpointRef = {
    "repo_id": "kabartay/fintfm-binary",
    "filename": "v4-cellattn-labels.pt",
    "revision": "f116bfd43a2b15c65ed3551ea8c38e3364629ddc",
    "sha256": "d231b062483c9261aae4c503ad5359db295fc7206fa0183f51763e4953c518d2",
}

#: The multiclass entry's checkpoint (``dr74-mc-s0``, ``max_classes=10``), Apache-2.0 and ungated
#: since 2026-11-17, pinned to a commit like :data:`BINARY_CHECKPOINT`.
MULTICLASS_CHECKPOINT: CheckpointRef = {
    "repo_id": "kabartay/fintfm-multiclass",
    "filename": "dr74-mc-s0.pt",
    "revision": "b292efa80fbebf759802a980bafe9b5bdcc626bd",
    "sha256": "14d5b4d561fbea6e35c7a30dc2644cb8f64f437c25d62da087ee960717cc8f4d",
}

#: The regression entry's checkpoint (``v4-binned48_3-control``, ``max_classes=10``), Apache-2.0
#: and ungated since 2026-11-17, pinned to a commit.
REGRESSION_CHECKPOINT: CheckpointRef = {
    "repo_id": "kabartay/fintfm-regression",
    "filename": "v4-binned48_3-control.pt",
    "revision": "787a3446685cdcd76eaf079637c50bc4c6fb5a04",
    "sha256": "31d5b3ef49e89bbe405fc85ab1d746a9fd6c3a1c5282e1770111902cc4af526c",
}


def sha256_of(path: str) -> str:
    """SHA-256 hex digest of a file, read in 1 MiB blocks.

    Args:
        path: The file to hash.

    Returns:
        The lower-case hex digest.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_checkpoint_ref(ref: Any, *, source: str) -> None:
    """Raise ``ValueError`` unless ``ref`` is a local path string or a well-formed reference.

    Args:
        ref: A local checkpoint path, or a reference dict with the keys ``repo_id``, ``filename``,
            ``revision`` and ``sha256``.
        source: Where the value came from, for the message.

    Raises:
        ValueError: ``ref`` is an empty string, is neither a string nor a dict, or is a dict with
            other keys.
    """
    if isinstance(ref, str):
        if not ref.strip():
            raise ValueError(f"{source}: the checkpoint path is empty")
        return
    if not isinstance(ref, dict):
        raise ValueError(  # noqa: TRY004
            f"{source}: a checkpoint is a local path or a reference dict, got {type(ref).__name__}"
        )
    keys = set(ref)
    if keys != _PUBLISHED_KEYS:
        raise ValueError(
            f"{source}: checkpoint reference keys {sorted(keys)} do not match a published reference "
            f"{sorted(_PUBLISHED_KEYS)}"
        )


def resolve_checkpoint(ref: Any, *, problem_type: str) -> str:
    """The local path of the checkpoint ``ref`` names, downloading and verifying it if published.

    A string is a local path and is returned unchanged. A published reference is fetched from the
    Hugging Face cache at its pinned commit and its SHA-256 is checked. ``huggingface_hub`` is imported
    here, not at module level, because it is not a core dependency.

    Args:
        ref: A path string, or a reference dict from :func:`validate_checkpoint_ref`.
        problem_type: ``binary``, ``multiclass`` or ``regression``, for the error message.

    Returns:
        The path of a checkpoint file.

    Raises:
        ImportError: ``ref`` is a published reference and ``huggingface_hub`` is not installed.
        ValueError: ``ref`` is malformed, or the downloaded file's SHA-256 differs.
    """
    validate_checkpoint_ref(ref, source=f"checkpoint_{problem_type}")
    if isinstance(ref, str):
        return ref
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            f"Fetching the published {problem_type} checkpoint {ref['repo_id']}/{ref['filename']} needs "
            "huggingface_hub (pip install huggingface_hub), or pass a local checkpoint path instead."
        ) from exc

    path = hf_hub_download(
        repo_id=ref["repo_id"], filename=ref["filename"], revision=ref["revision"]
    )
    actual = sha256_of(path)
    if actual != ref["sha256"]:
        raise ValueError(
            f"FinTFM {problem_type} checkpoint {ref['repo_id']}/{ref['filename']}@{ref['revision']} has "
            f"sha256 {actual}, expected {ref['sha256']}"
        )
    return path
