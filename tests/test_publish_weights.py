"""Dry-run and refusal paths of ``scripts/publish_weights.py``. No network, no real weights."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "publish_weights.py"


@pytest.fixture
def pw(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Load the script as a module with three tiny stand-in checkpoints.

    Returns:
        ``(module, runs_dir)``.
    """
    spec = importlib.util.spec_from_file_location("publish_weights", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["publish_weights"] = mod
    spec.loader.exec_module(mod)
    runs = tmp_path / "runs"
    runs.mkdir()
    fake = []
    for c in mod.CHECKPOINTS:
        data = c.key.encode() * 10
        (runs / c.relpath).parent.mkdir(parents=True, exist_ok=True)
        (runs / c.relpath).write_bytes(data)
        fake.append(
            type(c)(
                c.key,
                c.repo_id,
                c.relpath,
                c.filename,
                hashlib.sha256(data).hexdigest(),
                len(data),
                c.card,
                c.published,
            )
        )
    monkeypatch.setattr(mod, "CHECKPOINTS", tuple(fake))
    return mod, runs


def test_recorded_hashes_are_well_formed() -> None:
    """The real records carry 64-hex digests and the three expected keys."""
    spec = importlib.util.spec_from_file_location("pw_real", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pw_real"] = mod
    spec.loader.exec_module(mod)
    assert [c.key for c in mod.CHECKPOINTS] == ["binary", "multiclass", "regression"]
    for c in mod.CHECKPOINTS:
        assert len(c.sha256) == 64 and int(c.sha256, 16) >= 0
        assert (SCRIPT.parent.parent / c.card).is_file()


def test_dry_run_prints_and_exits_zero(pw, capsys: pytest.CaptureFixture[str]) -> None:
    """With no flags the script verifies hashes, prints the plan and returns 0."""
    mod, runs = pw
    assert mod.main(["--runs-dir", str(runs)], env={}) == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "kabartay/fintfm-multiclass" in out and "all hashes match" in out


def test_hash_mismatch_fails(pw, capsys: pytest.CaptureFixture[str]) -> None:
    """A changed file is reported and the exit code is 1."""
    mod, runs = pw
    (runs / mod.CHECKPOINTS[1].relpath).write_bytes(b"tampered")
    assert mod.main(["--runs-dir", str(runs)], env={}) == 1
    assert "multiclass" in capsys.readouterr().err


def test_missing_file_fails(pw) -> None:
    """A missing checkpoint exits 1."""
    mod, runs = pw
    (runs / mod.CHECKPOINTS[2].relpath).unlink()
    assert mod.main(["--runs-dir", str(runs)], env={}) == 1


def test_publish_refused_without_token(pw, capsys: pytest.CaptureFixture[str]) -> None:
    """--yes-publish with no HF_TOKEN is refused with exit 2 and uploads nothing."""
    mod, runs = pw
    assert mod.main(["--runs-dir", str(runs), "--yes-publish", "--licence", "X"], env={}) == 2
    assert "HF_TOKEN" in capsys.readouterr().err


def test_publish_refused_without_licence(pw, capsys: pytest.CaptureFixture[str]) -> None:
    """A token alone is not enough: the owner must name a licence."""
    mod, runs = pw
    code = mod.main(["--runs-dir", str(runs), "--yes-publish"], env={"HF_TOKEN": "t"})
    assert code == 2
    assert "--licence" in capsys.readouterr().err


def test_publish_refused_while_card_says_owner_decision(pw, monkeypatch) -> None:
    """With token and licence given, cards that still carry the placeholder block the upload."""
    mod, runs = pw

    def boom(*a, **k):
        raise AssertionError("upload must not be reached")

    monkeypatch.setattr(mod, "_upload", boom)
    # The real cards now carry the owner's licence (2026-11-16); the guard still has to hold
    # for any card that says the placeholder, so point the script at a copy that does.
    fake_root = runs.parent / "fake_repo"
    for c in mod.CHECKPOINTS:
        card = fake_root / c.card
        card.parent.mkdir(parents=True, exist_ok=True)
        card.write_text(f"LICENCE: {mod.PLACEHOLDER}\n")
    monkeypatch.setattr(mod, "REPO_ROOT", fake_root)
    code = mod.main(
        ["--runs-dir", str(runs), "--yes-publish", "--licence", "Apache-2.0"],
        env={"HF_TOKEN": "t"},
    )
    assert code == 2
