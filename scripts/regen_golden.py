"""Golden-output fingerprints: what the published checkpoint returns, pinned bit for bit (DR1.1).

Phase DR1 of ``openspec/changes/design-review-2026-10-16`` promises that no output may change
while the inference code is reorganised (DR1.3 configuration, DR1.4 one inference path). A
promise like that is only checkable against a record of what the outputs *were*, so this
module scores ``runs/v4-cellattn-labels.pt`` on a seeded prior table through every inference
lever the published results depend on and stores a SHA-256 of each output's raw bytes. A
1-ulp change anywhere in an output changes its hash; ``tests/test_golden_outputs.py`` asserts
equality and proves that by perturbing each output.

Two fixtures, because they can be checked in different places:

* ``tests/fixtures/golden_v4cellattn.npz`` -- the checkpoint's outputs. The weights are
  gitignored (``CLAUDE.md``), so this runs only on a machine that has them: locally, from
  ``scripts/pre-push``. Bit equality is meaningful only on the platform and library versions
  that recorded it, so the test skips with a rebaseline message when torch, numpy or the
  platform differ.
* ``tests/fixtures/golden_random_init.npz`` -- tiny seeded random-initialisation models (the
  classifier, the quantile regressor, the term structure). These need no checkpoint and run
  in CI. They also store the arrays themselves, so off the recording platform the test falls
  back to a tolerance check that still catches any behavioural change, though not a 1-ulp one.

**Regenerate deliberately**, never to make a red test green: a mismatch means an output moved,
and DR1 says none may. Rebaseline only for a torch/numpy upgrade or a change that is meant to
move outputs (DR1.4's ``predict_term_structure`` seed fix is the one named so far), and say so
in the commit.

Usage:
    uv run python scripts/regen_golden.py                      # both fixtures
    uv run python scripts/regen_golden.py --only random-init   # CI fixture, no checkpoint
    uv run python scripts/regen_golden.py --checkpoint /path/to/v4-cellattn-labels.pt

The checkpoint is found, in order, at ``--checkpoint``, ``$FINTFM_GOLDEN_CHECKPOINT``, this
checkout's ``runs/`` and the main checkout's ``runs/`` (so a git worktree finds it too), and is
refused unless its SHA-256 is :data:`CHECKPOINT_SHA256`.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import subprocess
import sys
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import torch

from fintfm.inference.classifier import FinancialTFMClassifier
from fintfm.inference.quantile_regressor import FinancialTFMQuantileRegressor
from fintfm.modeling.hazard import CENSORED
from fintfm.modeling.model import FinancialTFM, ModelConfig
from fintfm.prior.financial import sample_financial_task

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures"
CHECKPOINT_FIXTURE = FIXTURES / "golden_v4cellattn.npz"
RANDOM_INIT_FIXTURE = FIXTURES / "golden_random_init.npz"

CHECKPOINT_NAME = "v4-cellattn-labels.pt"
#: The published checkpoint, as cited in ``docs/results/FINDINGS.md``. Anything else is refused.
CHECKPOINT_SHA256 = "d231b062483c9261aae4c503ad5359db295fc7206fa0183f51763e4953c518d2"
#: Environment variable naming the checkpoint when it is not under a ``runs/`` directory.
CHECKPOINT_ENV = "FINTFM_GOLDEN_CHECKPOINT"

#: The golden table: one financial-prior task, the first ``N_CONTEXT`` rows as the training
#: table and the rest as queries. Seed 2 gives a 2.1% positive rate and 13% missing cells, so
#: the low-base-rate correction and the NaN path are both exercised.
TABLE_SEED = 2
N_CONTEXT = 2000
N_QUERY = 512
N_FEATURES = 136
#: ``FinancialTFMClassifier.transform_representation``'s stages; a test pins this to the
#: method's own type hint, so a new stage cannot go unrecorded.
STAGES = ("encode_rows", "encoder_mid", "pre_head")
#: Intra-op threads every output is computed with, so the record does not depend on the
#: machine's core count. Restored afterwards.
N_THREADS = 4
#: Values kept per output, evenly strided, so a mismatch can report how far it moved.
N_SAMPLE = 64


def sha256_bytes(data: bytes) -> str:
    """Hex SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file, read in 1 MiB blocks.

    Args:
        path: File to hash.

    Returns:
        The digest as 64 hex characters.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha256(arr: np.ndarray) -> str:
    """SHA-256 of an array's raw bytes, with its dtype and shape folded in.

    Folding the dtype and shape in means a float64 output that replaced a float32 one, or a
    reshaped one with the same bytes, is a different fingerprint rather than a collision.

    Args:
        arr: Any numeric array. It is hashed as it is, never cast.

    Returns:
        The digest as 64 hex characters.
    """
    arr = np.ascontiguousarray(arr)
    header = f"{arr.dtype.str}{arr.shape}".encode()
    return sha256_bytes(header + arr.tobytes())


def strided_sample(arr: np.ndarray, n: int = N_SAMPLE) -> np.ndarray:
    """``n`` evenly strided values of ``arr`` (flattened), first and last included.

    Args:
        arr: Any array.
        n: Values to keep.

    Returns:
        ``(min(n, arr.size),)`` values in ``arr``'s own dtype.
    """
    flat = np.ascontiguousarray(arr).ravel()
    idx = np.unique(np.linspace(0, flat.size - 1, num=min(n, flat.size)).round().astype(int))
    return flat[idx]


def environment() -> dict[str, str]:
    """The library versions and platform a bit-exact record is valid for.

    Returns:
        ``torch``, ``numpy`` and ``platform`` (system and machine, e.g. ``Darwin-arm64``).
    """
    return {
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": f"{platform.system()}-{platform.machine()}",
    }


@contextlib.contextmanager
def pinned_threads(n: int = N_THREADS) -> Iterator[None]:
    """Run with ``n`` torch intra-op threads, restoring the caller's setting afterwards.

    Args:
        n: Thread count.
    """
    before = torch.get_num_threads()
    torch.set_num_threads(n)
    try:
        yield
    finally:
        torch.set_num_threads(before)


def golden_table() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The seeded prior table every checkpoint output is computed on.

    Returns:
        ``(X_context, y_context, X_query)``: ``(2000, 136)`` float32 training rows, their
        ``(2000,)`` binary labels, and ``(512, 136)`` float32 query rows.
    """
    task = sample_financial_task(
        np.random.default_rng(TABLE_SEED),
        N_CONTEXT + N_QUERY,
        max_features=N_FEATURES,
        min_features=N_FEATURES,
    )
    X = task.X.astype(np.float32)
    return X[:N_CONTEXT], task.y[:N_CONTEXT].astype(np.int64), X[N_CONTEXT:]


def table_sha256(*arrays: np.ndarray) -> str:
    """One digest over the golden table, so a prior change is told apart from a model change.

    Args:
        arrays: The table's arrays, in a fixed order.

    Returns:
        The digest as 64 hex characters.
    """
    return sha256_bytes(b"".join(array_sha256(a).encode() for a in arrays))


def _seeded(fn: Callable[[], np.ndarray]) -> Callable[[], np.ndarray]:
    """Wrap a scenario so it starts from torch seed 0, whatever ran before it.

    Nothing on the inference path should read torch's global generator; seeding anyway means
    a scenario's output cannot depend on which scenarios ran first.
    """

    def run() -> np.ndarray:
        """Seed, then run the scenario."""
        torch.manual_seed(0)
        return fn()

    return run


def classifier_scenarios(
    model: FinancialTFM,
    X: np.ndarray,
    y: np.ndarray,
    Xq: np.ndarray,
    retrieval_context: int,
    finetune_context: int,
    finetune_query: int,
) -> dict[str, Callable[[], np.ndarray]]:
    """Every classifier lever the fingerprint covers, as named zero-argument scenarios.

    Shared by the checkpoint and the random-initialisation fixtures, so the two cover the same
    levers. Everything not named here is at the classifier's default.

    Args:
        model: The network; never modified (fine-tuning works on a copy).
        X: ``(n, F)`` training rows.
        y: ``(n,)`` labels.
        Xq: ``(m, F)`` query rows.
        retrieval_context: ``max_context`` for the retrieval scenario. Below ``n``, or
            retrieval would return the whole pool and test nothing.
        finetune_context: ``finetune_context`` for the fine-tuning scenario.
        finetune_query: ``finetune_query`` for the fine-tuning scenario.

    Returns:
        Scenario name to a callable returning its output array.
    """

    def proba(**kwargs: object) -> Callable[[], np.ndarray]:
        """A scenario that fits with ``kwargs`` and returns ``predict_proba``."""

        def run() -> np.ndarray:
            """Fit and score."""
            clf = FinancialTFMClassifier(model, **kwargs)  # type: ignore[arg-type]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                clf.fit(X, y)
            return clf.predict_proba(Xq)

        return run

    def representation(stage: str) -> Callable[[], np.ndarray]:
        """A scenario returning ``transform_representation`` at ``stage``."""

        def run() -> np.ndarray:
            """Fit and encode."""
            clf = FinancialTFMClassifier(model).fit(X, y)
            return clf.transform_representation(Xq, stage=stage)  # type: ignore[arg-type]

        return run

    scenarios: dict[str, Callable[[], np.ndarray]] = {
        "predict_proba": proba(),
        "ensemble2_label_swap": proba(n_ensemble=2, ensemble_label_swap=True),
        "retrieval": proba(context_strategy="retrieval", max_context=retrieval_context),
        "finetune3": proba(
            finetune_steps=3,
            finetune_seed=7,
            finetune_context=finetune_context,
            finetune_query=finetune_query,
        ),
    }
    for stage in STAGES:
        scenarios[f"representation_{stage}"] = representation(stage)
    return {name: _seeded(fn) for name, fn in scenarios.items()}


def term_structure_scenario(
    model: FinancialTFM, X: np.ndarray, y: np.ndarray, Xq: np.ndarray
) -> Callable[[], np.ndarray]:
    """``predict_term_structure`` at the classifier's defaults, as a scenario.

    Args:
        model: A network with a hazard head (else the scenario raises ``RuntimeError``).
        X: ``(n, F)`` training rows.
        y: ``(n,)`` binary labels.
        Xq: ``(m, F)`` query rows.

    Returns:
        A callable returning the ``(m, n_horizons)`` cumulative PD.
    """

    def run() -> np.ndarray:
        """Fit and predict the term structure."""
        return FinancialTFMClassifier(model).fit(X, y).predict_term_structure(Xq)

    return _seeded(run)


def load_checkpoint(path: Path) -> FinancialTFM:
    """Load the published checkpoint after checking it is that checkpoint.

    Args:
        path: The checkpoint file.

    Returns:
        The model, on CPU, in eval mode.

    Raises:
        ValueError: If the file's SHA-256 is not :data:`CHECKPOINT_SHA256`.
    """
    digest = sha256_file(path)
    if digest != CHECKPOINT_SHA256:
        raise ValueError(
            f"{path} has SHA-256 {digest}, not the published checkpoint's {CHECKPOINT_SHA256}"
        )
    return FinancialTFM.load(str(path), map_location="cpu")


def locate_checkpoint(explicit: str | None = None) -> Path | None:
    """Find the published checkpoint without assuming where this checkout lives.

    Args:
        explicit: A path given on the command line; wins when set.

    Returns:
        The first existing candidate among ``explicit``, ``$FINTFM_GOLDEN_CHECKPOINT``,
        ``<this checkout>/runs/`` and ``<main checkout>/runs/``, or ``None``. The main
        checkout is found through git's common directory, which is what makes a worktree
        (whose own ``runs/`` is empty, the directory being gitignored) find the file.
    """
    candidates: list[Path] = []
    for given in (explicit, os.environ.get(CHECKPOINT_ENV)):
        if given:
            candidates.append(Path(given).expanduser())
    candidates.append(REPO / "runs" / CHECKPOINT_NAME)
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if common:
            candidates.append(Path(common).parent / "runs" / CHECKPOINT_NAME)
    return next((p for p in candidates if p.is_file()), None)


def checkpoint_outputs(model: FinancialTFM) -> tuple[dict[str, np.ndarray], str | None]:
    """Score the golden table through every covered lever of the published checkpoint.

    Args:
        model: The published checkpoint, from :func:`load_checkpoint`.

    Returns:
        ``(outputs, term_structure_refusal)``: scenario name to output array, and the
        ``RuntimeError`` message ``predict_term_structure`` raised, or ``None`` when the
        checkpoint has a hazard head and its term structure is among the outputs instead.
    """
    X, y, Xq = golden_table()
    scenarios = classifier_scenarios(
        model, X, y, Xq, retrieval_context=256, finetune_context=256, finetune_query=64
    )
    refusal = None
    if model.hazard is None:
        try:
            term_structure_scenario(model, X, y, Xq)()
        except RuntimeError as exc:
            refusal = str(exc)
    else:
        scenarios["term_structure"] = term_structure_scenario(model, X, y, Xq)
    with pinned_threads():
        return {name: fn() for name, fn in scenarios.items()}, refusal


def _tiny_task(seed: int, n_rows: int, n_features: int, n_horizons: int | None = None):
    """A small seeded financial-prior task for the random-initialisation fingerprints."""
    return sample_financial_task(
        np.random.default_rng(seed),
        n_rows,
        max_features=n_features,
        min_features=n_features,
        n_horizons=n_horizons,
    )


def random_init_outputs() -> dict[str, np.ndarray]:
    """Outputs of tiny seeded random-initialisation models, for the CI fingerprint.

    No checkpoint is read. Each model is built right after ``torch.manual_seed`` so its weights
    are a pure function of the seed, and it is small enough that the whole set runs in about a
    second on CPU. Covers the classifier (the same levers as the checkpoint fixture), the
    quantile regressor and the term structure.

    Returns:
        Output name to array.
    """
    tiny = {
        "max_features": 8,
        "d_cell": 8,
        "d_model": 16,
        "n_heads": 2,
        "n_col_layers": 1,
        "n_layers": 2,
        "d_ff": 32,
        "column_id_dim": 4,
    }
    outputs: dict[str, np.ndarray] = {}
    with pinned_threads():
        task = _tiny_task(seed=1, n_rows=160, n_features=8)
        X, Xq = task.X[:128].astype(np.float32), task.X[128:].astype(np.float32)
        y = task.y[:128].astype(np.int64)
        torch.manual_seed(0)
        clf_model = FinancialTFM(
            ModelConfig(max_classes=2, n_cell_blocks=1, cell_labels=True, **tiny)
        )
        scenarios = classifier_scenarios(
            clf_model, X, y, Xq, retrieval_context=32, finetune_context=48, finetune_query=16
        )
        outputs.update({f"classifier_{name}": fn() for name, fn in scenarios.items()})

        torch.manual_seed(1)
        q_model = FinancialTFM(
            ModelConfig(max_classes=2, head_type="quantile", n_quantiles=9, **tiny)
        )
        target = np.random.default_rng(3).normal(size=160) + np.nan_to_num(task.X[:, 0])
        torch.manual_seed(0)
        reg = FinancialTFMQuantileRegressor(q_model, max_context=96).fit(X, target[:128])
        outputs["quantile_grid"] = reg.predict_quantile_grid(Xq)

        hz = _tiny_task(seed=4, n_rows=160, n_features=8, n_horizons=3)
        assert hz.period is not None
        y_hz = (hz.period != CENSORED).astype(np.int64)
        torch.manual_seed(2)
        ts_model = FinancialTFM(ModelConfig(max_classes=2, n_horizons=3, **tiny))
        outputs["term_structure"] = term_structure_scenario(
            ts_model, hz.X[:128].astype(np.float32), y_hz[:128], hz.X[128:].astype(np.float32)
        )()
    return outputs


def write_fixture(
    path: Path,
    outputs: dict[str, np.ndarray],
    meta: dict[str, object],
    keep_arrays: bool,
) -> None:
    """Write one fixture: per output its hash, shape, dtype and a strided sample.

    Args:
        path: Destination ``.npz``.
        outputs: Output name to array.
        meta: JSON-serialisable metadata (environment, checkpoint and table digests).
        keep_arrays: Also store each array in full; only for the tiny random-init outputs.
    """
    data: dict[str, np.ndarray] = {"meta": np.array(json.dumps(meta, sort_keys=True))}
    for name, arr in outputs.items():
        data[f"{name}.sha256"] = np.array(array_sha256(arr))
        data[f"{name}.sample"] = strided_sample(arr)
        if keep_arrays:
            data[f"{name}.array"] = np.ascontiguousarray(arr)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def read_fixture(path: Path) -> tuple[dict[str, object], dict[str, dict[str, np.ndarray]]]:
    """Read a fixture written by :func:`write_fixture`.

    Args:
        path: The ``.npz``.

    Returns:
        ``(meta, outputs)``, ``outputs`` mapping each name to its ``sha256``, ``sample`` and,
        when stored, ``array`` entries.
    """
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        outputs: dict[str, dict[str, np.ndarray]] = {}
        for key in z.files:
            if key == "meta":
                continue
            name, field = key.rsplit(".", 1)
            outputs.setdefault(name, {})[field] = z[key]
    return meta, outputs


def main(argv: list[str] | None = None) -> int:
    """Regenerate one or both fixtures.

    Args:
        argv: Command-line arguments; ``sys.argv[1:]`` when ``None``.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--only", choices=("checkpoint", "random-init"), default=None)
    parser.add_argument("--checkpoint", default=None, help=f"path to {CHECKPOINT_NAME}")
    args = parser.parse_args(argv)

    if args.only != "checkpoint":
        outputs = random_init_outputs()
        write_fixture(RANDOM_INIT_FIXTURE, outputs, {"env": environment()}, keep_arrays=True)
        print(f"wrote {RANDOM_INIT_FIXTURE.relative_to(REPO)}: {len(outputs)} outputs")
    if args.only != "random-init":
        path = locate_checkpoint(args.checkpoint)
        if path is None:
            print(f"{CHECKPOINT_NAME} not found; pass --checkpoint or set {CHECKPOINT_ENV}")
            return 1
        model = load_checkpoint(path)
        outputs, refusal = checkpoint_outputs(model)
        meta = {
            "env": environment(),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "table_sha256": table_sha256(*golden_table()),
            "term_structure_refusal": refusal,
            "threads": N_THREADS,
        }
        write_fixture(CHECKPOINT_FIXTURE, outputs, meta, keep_arrays=False)
        print(f"wrote {CHECKPOINT_FIXTURE.relative_to(REPO)}: {len(outputs)} outputs")
        for name, arr in outputs.items():
            print(f"  {name:32s} {arr.dtype} {arr.shape} {array_sha256(arr)[:16]}")
        if refusal is not None:
            print(f"  predict_term_structure refused: {refusal}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
