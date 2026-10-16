"""What produced an artefact, recorded beside it (design review DR1.2).

Why this exists
---------------
A checkpoint used to carry its architecture, its weights and the objectives it was trained on,
and nothing else. The prior it was trained under, the training settings, the commit, the
library versions and the number of tasks it had seen all lived in a shell history or a
FINDINGS paragraph, so a question such as "what financial share did this checkpoint actually
see?" (``docs/results/FINDINGS.md`` §166) could only be answered from memory. These helpers
build the record that :meth:`fintfm.modeling.model.FinancialTFM.save`, the training loop and
the V4 protocol harness now write.

Every record here is JSON-safe (nested ``dict``/``list``/``str``/``int``/``float``/``bool``/
``None``), because a checkpoint is read with ``torch.load(weights_only=True)``, which refuses
arbitrary objects, and because a run JSON must serialise.

Nothing here touches a model's numbers: a record is written, never read back by inference.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

#: Bumped on any incompatible change to the shape of :func:`environment_record`'s output, so a
#: reader can tell which fields to expect.
PROVENANCE_FORMAT = 1

#: The installed package's own directory: the git state reported is the state of the code that
#: is running, whatever directory the command was launched from.
_PACKAGE_DIR = Path(__file__).resolve().parent


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file's bytes, read in chunks so a large checkpoint is not held in memory.

    Args:
        path: File to hash.
        chunk_size: Bytes per read.

    Returns:
        The lowercase hexadecimal digest.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(directory: Path, *args: str) -> str | None:
    """Run one git command in ``directory``; ``None`` on any failure, never an exception."""
    try:
        out = subprocess.run(
            ["git", "-C", str(directory), *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout


def git_state(directory: str | Path | None = None) -> dict[str, str | bool | None]:
    """The commit and dirty flag of the repository holding ``directory``, best effort.

    Args:
        directory: Any path inside the repository. Defaults to this package's own directory,
            so an installed (non-editable) package reports ``None`` rather than the state of
            whatever repository the caller happens to be in.

    Returns:
        ``{"commit": sha or None, "dirty": bool or None}``. ``dirty`` counts changes to
        tracked files only, as ``git describe --dirty`` does; an untracked scratch file does
        not make every checkpoint dirty. Both are ``None`` outside a repository or without git.
    """
    where = _PACKAGE_DIR if directory is None else Path(directory)
    sha = _git(where, "rev-parse", "HEAD")
    if sha is None:
        return {"commit": None, "dirty": None}
    status = _git(where, "status", "--porcelain", "--untracked-files=no")
    return {"commit": sha.strip(), "dirty": None if status is None else bool(status.strip())}


def package_versions() -> dict[str, str]:
    """The ``fintfm`` and ``torch`` versions this process is running.

    Returns:
        ``{"fintfm": ..., "torch": ...}``.
    """
    import torch

    import fintfm

    # str(): torch.__version__ is a TorchVersion, which weights_only loading refuses
    return {"fintfm": str(fintfm.__version__), "torch": str(torch.__version__)}


def environment_record() -> dict[str, Any]:
    """Commit, dirty flag and versions: the part of every record that describes the code.

    Returns:
        ``{"format", "commit", "dirty", "versions"}``.
    """
    return json_safe({"format": PROVENANCE_FORMAT, **git_state(), "versions": package_versions()})


def json_safe(value: Any) -> Any:
    """Convert ``value`` into nested JSON types, recursively.

    Tuples become lists, mapping keys become strings, NumPy scalars and arrays become Python
    numbers and lists, and paths become strings. Anything else that JSON cannot carry is
    recorded as its ``repr`` rather than dropped, so a record never silently loses a field.

    Args:
        value: Any value, typically ``dataclasses.asdict`` of a configuration.

    Returns:
        A value ``json.dumps`` accepts and ``torch.load(weights_only=True)`` reads back.
    """
    if value is None:
        return value
    # Subclasses are cast to the base type: ``torch.__version__`` is a ``str`` subclass and
    # ``np.float64`` a ``float`` one, and ``weights_only`` loading refuses both.
    for base in (bool, int, float, str):
        if isinstance(value, base):
            return base(value)
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    tolist = getattr(value, "tolist", None)  # numpy scalars and arrays, torch tensors
    if callable(tolist):
        return json_safe(tolist())
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError):
        return repr(value)


def estimator_params(estimator: Any) -> dict[str, Any]:
    """``estimator.get_params(deep=True)``, JSON-safe, with model and estimator objects dropped.

    The classifier's ``model`` argument is the network itself, and a ``large_n`` wrapper's
    ``estimator`` is the inner classifier, whose own settings ``deep=True`` already lists
    under ``estimator__*``. Neither belongs in a JSON record; the checkpoint is identified by
    its SHA-256 instead.

    Args:
        estimator: A scikit-learn-style estimator with ``get_params``.

    Returns:
        Every remaining parameter, by name, sorted.
    """
    import torch
    from sklearn.base import BaseEstimator

    params = estimator.get_params(deep=True)
    return {
        k: json_safe(v)
        for k, v in sorted(params.items())
        if not isinstance(v, (torch.nn.Module, BaseEstimator))
    }


def read_checkpoint_provenance(path: str | Path) -> dict[str, Any] | None:
    """The provenance a checkpoint recorded, or ``None`` for one written before DR1.2.

    Args:
        path: A checkpoint written by :meth:`fintfm.modeling.model.FinancialTFM.save`.

    Returns:
        The ``provenance`` mapping, or ``None`` when the checkpoint has none.
    """
    import torch

    ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    return ckpt.get("provenance")
