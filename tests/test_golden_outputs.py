"""Golden-output fingerprints: no inference output may move without anyone noticing (DR1.1).

Phase DR1 of ``openspec/changes/design-review-2026-10-16`` reorganises the inference code under
the promise that no output changes. These tests are what make that promise checkable. The
scenarios and the fixture format live in ``scripts/regen_golden.py``, which is also the only
sanctioned way to rebaseline.

* **The published checkpoint** (``runs/v4-cellattn-labels.pt``, gitignored) is scored on a
  seeded 2,000 x 136 prior table plus 512 queries through every lever the published results
  depend on, and each output's raw bytes must hash to the recorded SHA-256: exact equality,
  not a tolerance. Skipped when the checkpoint is absent (CI) and when torch, numpy or the
  platform differ from the recording, since bit equality is not promised across them. Run
  from ``scripts/pre-push``.
* **Tiny random-initialisation models** of the classifier, the quantile regressor and the term
  structure need no checkpoint and run everywhere. Exact on the recording environment; off it,
  a tolerance check against the stored arrays, which catches a behavioural change but not a
  1-ulp one.

The DR1.1 verification -- the test fails when any one output is perturbed by 1 ulp -- is
asserted rather than described: every recorded output, and on the checkpoint every freshly
computed one, is nudged by one ulp at several positions and must then fail the comparison.
"""

from __future__ import annotations

import sys
import typing
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import regen_golden as golden

from fintfm.inference.classifier import FinancialTFMClassifier

REBASELINE = (
    "rebaseline deliberately with `uv run python scripts/regen_golden.py` after checking why "
    "outputs moved, and say so in the commit"
)

_ri_meta, _ri_golden = golden.read_fixture(golden.RANDOM_INIT_FIXTURE)
_ck_meta, _ck_golden = golden.read_fixture(golden.CHECKPOINT_FIXTURE)


def matches(recorded: dict[str, np.ndarray], actual: np.ndarray) -> bool:
    """Whether ``actual`` is bit-for-bit the recorded output (dtype and shape included).

    Args:
        recorded: The fixture's entries for one output.
        actual: The freshly computed output.

    Returns:
        True only on an identical SHA-256 of the raw bytes.
    """
    return golden.array_sha256(actual) == str(recorded["sha256"])


def describe_mismatch(recorded: dict[str, np.ndarray], actual: np.ndarray) -> str:
    """How far a mismatched output moved, read off the stored strided sample.

    Args:
        recorded: The fixture's entries for one output.
        actual: The freshly computed output.

    Returns:
        A one-line description for an assertion message.
    """
    sample = recorded["sample"]
    now = golden.strided_sample(actual)
    if now.shape != sample.shape or now.dtype != sample.dtype:
        return f"sample shape/dtype {now.shape} {now.dtype} vs recorded {sample.shape} {sample.dtype}"
    diff = np.abs(now.astype(np.float64) - sample.astype(np.float64))
    return (
        f"{int((diff > 0).sum())}/{diff.size} sampled values differ, max |diff| {diff.max():.3e}"
    )


def one_ulp_variants(arr: np.ndarray) -> list[np.ndarray]:
    """Copies of ``arr`` with exactly one element moved by one ulp, at several positions.

    The first, middle and last elements, each nudged up and down, so a comparison that only
    looked at part of the array, or rounded, would let one of them through.

    Args:
        arr: A floating-point array.

    Returns:
        Six perturbed copies, each differing from ``arr`` in exactly one element.
    """
    out = []
    for pos in (0, arr.size // 2, arr.size - 1):
        for direction in (np.inf, -np.inf):
            nudged = np.array(arr, copy=True)
            flat = nudged.reshape(-1)
            flat[pos] = np.nextafter(flat[pos], direction)
            assert np.count_nonzero(nudged != arr) == 1
            out.append(nudged)
    return out


def environment_mismatch(meta: dict[str, object]) -> str | None:
    """Which of torch, numpy and platform differ from the recording, or ``None``."""
    recorded = meta["env"]
    assert isinstance(recorded, dict)
    now = golden.environment()
    diffs = [f"{k} {now[k]} (recorded {recorded[k]})" for k in now if now[k] != recorded.get(k)]
    return ", ".join(diffs) or None


# --- random initialisation: runs everywhere, CI included ------------------------------------


@pytest.fixture(scope="module")
def random_init_outputs() -> dict[str, np.ndarray]:
    """The tiny random-initialisation outputs, computed once for the module."""
    return golden.random_init_outputs()


def test_random_init_fixture_covers_every_output(random_init_outputs):
    assert sorted(random_init_outputs) == sorted(_ri_golden)


@pytest.mark.parametrize("name", sorted(_ri_golden))
def test_random_init_fingerprint(name, random_init_outputs):
    recorded, actual = _ri_golden[name], random_init_outputs[name]
    stored = recorded["array"]
    assert actual.dtype == stored.dtype and actual.shape == stored.shape
    mismatch = environment_mismatch(_ri_meta)
    if mismatch is None:
        assert matches(recorded, actual), (
            f"{name} moved: {describe_mismatch(recorded, actual)}; {REBASELINE}"
        )
    else:
        # bit equality is not promised across BLAS builds and CPUs; a real change in what the
        # model computes moves a random-init output by far more than this
        if "retrieval" in name:
            # nearest-neighbour retrieval turns a last-digit distance difference into a different
            # neighbour set when two candidates nearly tie, so off the recording platform a few
            # query rows legitimately move (CI on Linux: 2 of 32 rows, by up to 0.007). Hold the
            # rest exactly and the moved rows to a coarse bound; a real change moves most rows.
            close = np.isclose(actual, stored, rtol=1e-4, atol=1e-5).all(axis=-1)
            assert close.mean() >= 0.9, f"{name}: only {close.mean():.0%} of rows held ({mismatch})"
            np.testing.assert_allclose(actual, stored, atol=0.05, err_msg=f"{name} moved ({mismatch})")
            return
        np.testing.assert_allclose(
            actual, stored, rtol=1e-4, atol=1e-5, err_msg=f"{name} moved ({mismatch})"
        )


@pytest.mark.parametrize("name", sorted(_ri_golden))
def test_one_ulp_perturbation_fails_the_fingerprint(name):
    """DR1.1's verification, on outputs available everywhere: one ulp anywhere is caught."""
    recorded = _ri_golden[name]
    stored = recorded["array"]
    assert matches(recorded, stored)  # the stored array is the one that was hashed
    for nudged in one_ulp_variants(stored):
        assert not matches(recorded, nudged)


def test_representation_stages_are_all_recorded():
    """A new ``transform_representation`` stage must be added to the fingerprint."""
    hints = typing.get_type_hints(FinancialTFMClassifier.transform_representation)
    assert typing.get_args(hints["stage"]) == golden.STAGES


# --- the published checkpoint: local only, from scripts/pre-push ----------------------------


@pytest.fixture(scope="module")
def checkpoint_outputs() -> tuple[dict[str, np.ndarray], str | None]:
    """The published checkpoint's outputs, computed once; skips when they cannot be compared."""
    path = golden.locate_checkpoint()
    if path is None:
        pytest.skip(
            f"{golden.CHECKPOINT_NAME} not found (gitignored); set {golden.CHECKPOINT_ENV} "
            "or keep it in the main checkout's runs/"
        )
    mismatch = environment_mismatch(_ck_meta)
    if mismatch is not None:
        pytest.skip(f"golden outputs were recorded on another environment: {mismatch}; {REBASELINE}")
    # a wrong file under the right name fails here rather than skipping: it is not absent
    model = golden.load_checkpoint(path)
    table = golden.golden_table()
    assert golden.table_sha256(*table) == _ck_meta["table_sha256"], (
        "the golden table changed, so the prior's output moved, not the model's; check the "
        f"prior change was intended, then {REBASELINE}"
    )
    return golden.checkpoint_outputs(model)


def test_checkpoint_fixture_is_for_the_published_checkpoint():
    assert _ck_meta["checkpoint_sha256"] == golden.CHECKPOINT_SHA256


def test_checkpoint_fixture_covers_every_output(checkpoint_outputs):
    outputs, _ = checkpoint_outputs
    assert sorted(outputs) == sorted(_ck_golden)


@pytest.mark.parametrize("name", sorted(_ck_golden))
def test_checkpoint_fingerprint(name, checkpoint_outputs):
    outputs, _ = checkpoint_outputs
    recorded, actual = _ck_golden[name], outputs[name]
    assert matches(recorded, actual), (
        f"{name} moved: {describe_mismatch(recorded, actual)}; {REBASELINE}"
    )
    # and the comparison is sharp enough to see one ulp in this very output
    for nudged in one_ulp_variants(actual):
        assert not matches(recorded, nudged)


def test_checkpoint_term_structure_is_refused(checkpoint_outputs):
    """The published checkpoint has no hazard head, so ``predict_term_structure`` must raise.

    Recorded rather than assumed: if a future fixture is recorded on a hazard checkpoint, its
    term structure is fingerprinted like any other output instead.
    """
    _, refusal = checkpoint_outputs
    if _ck_meta["term_structure_refusal"] is None:
        assert "term_structure" in _ck_golden
    else:
        assert refusal == _ck_meta["term_structure_refusal"]
        assert "no hazard head" in refusal
