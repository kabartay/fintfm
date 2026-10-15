"""Company-level comparison of V4FinBench runs, including runs saved before companies were (AU2.1).

`scripts/compare_v4_scores.py --cluster` resamples whole companies, and for runs whose npz lacks
`company_test` it reads `scripts/v4_fold_companies.py`'s reconstruction. A reconstruction that
silently misaligned companies with rows would produce a plausible interval on the wrong
clusters, so the reconstruction is checked against the run's saved labels and the comparison
refuses one that does not match.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compare_v4_scores import compare
from v4_fold_companies import reconstruct, verify


def _panel(n_companies=150, rows_each=5, seed=0):
    """A small panel: a few countries, companies with several rows each, a shared label."""
    rng = np.random.RandomState(seed)
    company = np.repeat(np.array([f"firm{i:03d}" for i in range(n_companies)], dtype=object), rows_each)
    country = np.repeat(rng.randint(0, 3, size=n_companies), rows_each)
    y = np.repeat((rng.rand(n_companies) < 0.3).astype(np.int64), rows_each)
    return y, country, company


def _write_run(d: Path, rebuilt, seed, with_companies=False):
    """A fake `fintfm-v4protocol` output on the rebuilt folds' rows."""
    rng = np.random.default_rng(seed)
    d.mkdir()
    for k, f in rebuilt.items():
        y = f["y_true"]
        extra = {"company_test": f["company_test"]} if with_companies else {}
        np.savez(d / f"predictions_fold{k}.npz", y_true=y, y_val=f["y_val"],
                 pred_fintfm=y * 0.5 + rng.random(len(y)), **extra)
    return d


def test_reconstruction_is_verified_against_saved_labels(tmp_path):
    rebuilt = reconstruct(*_panel())
    run = _write_run(tmp_path / "run", rebuilt, seed=0)
    assert verify(run, rebuilt) == [0, 1, 2, 3, 4]
    # a run on other rows (labels flipped in one fold) is refused, not silently paired
    bad = tmp_path / "bad"
    bad.mkdir()
    for k, f in rebuilt.items():
        y = 1 - f["y_true"] if k == 3 else f["y_true"]
        np.savez(bad / f"predictions_fold{k}.npz", y_true=y, pred_fintfm=np.zeros(len(y)))
    with pytest.raises(SystemExit, match="fold3"):
        verify(bad, rebuilt)


def test_every_reconstructed_test_company_lies_in_one_fold():
    rebuilt = reconstruct(*_panel())
    seen = [set(f["company_test"].tolist()) for f in rebuilt.values()]
    for i in range(5):
        for j in range(i + 1, 5):
            assert not seen[i] & seen[j]


def test_cluster_comparison_reads_companies_from_any_source(tmp_path):
    rebuilt = reconstruct(*_panel())
    ctrl = _write_run(tmp_path / "ctrl", rebuilt, seed=1)
    arm = _write_run(tmp_path / "arm", rebuilt, seed=2)
    with pytest.raises(SystemExit, match="--companies"):
        compare(ctrl, arm, n_boot=200, cluster=True)

    companies = tmp_path / "companies"
    companies.mkdir()
    for k, f in rebuilt.items():
        np.savez(companies / f"companies_fold{k}.npz", **f)
    rows = compare(ctrl, arm, n_boot=200, cluster=True, companies=companies)
    assert [r["fold"] for r in rows] == [0, 1, 2, 3, 4]
    assert all(np.isfinite(r["holm_company"]) for r in rows)

    # a run that saved its own companies needs no reconstruction, and gives the same answer
    arm_new = _write_run(tmp_path / "arm_new", rebuilt, seed=2, with_companies=True)
    same = compare(ctrl, arm_new, n_boot=200, cluster=True)
    assert [r["ci_company"] for r in same] == [r["ci_company"] for r in rows]

    # without --cluster the output is the row bootstrap alone
    plain = compare(ctrl, arm, n_boot=200)
    assert "ci_company" not in plain[0]
    assert [r["ci"] for r in plain] == [r["ci"] for r in rows]


def test_cluster_comparison_refuses_companies_from_other_rows(tmp_path):
    rebuilt = reconstruct(*_panel())
    other = reconstruct(*_panel(seed=5))
    ctrl = _write_run(tmp_path / "ctrl", rebuilt, seed=1)
    arm = _write_run(tmp_path / "arm", rebuilt, seed=2)
    companies = tmp_path / "companies"
    companies.mkdir()
    for k, f in other.items():
        np.savez(companies / f"companies_fold{k}.npz", **f)
    with pytest.raises(SystemExit, match="do not match"):
        compare(ctrl, arm, n_boot=200, cluster=True, companies=companies)
