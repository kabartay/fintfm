"""Reconstruct each fold's test companies for runs saved before the npz carried them (AU2.1).

`predictions_fold<k>.npz` gained ``company_test`` on 2026-10-16. Every earlier run holds only
labels and predictions, so a company-level bootstrap of §159-§164 needs the companies from
elsewhere. The fold assignment is deterministic (``v4_protocol.build_fold_assignments`` on the
parquet's row order, seed 42), so this rebuilds it with the protocol's own functions, codes the
companies with ``v4_protocol.company_codes``, and **refuses to write anything unless the
reconstructed labels equal every given run's saved ``y_true`` (and ``y_val`` where saved)**,
fold by fold.

Reads only the three grouping columns, so it needs well under the memory of a scoring run.
Full-panel runs only: a ``max_rows`` subsample changes the row set and is refused.

Usage:
    uv run python scripts/v4_fold_companies.py --root data/cache/v4finbench \\
        --out runs/audit-cluster/companies \\
        runs/task48-6-kaggle/control_rescore.json runs/ft/B_lr1e5 runs/ft/A_lr1e5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fintfm.experiments.v4_protocol import (
    FOLD_SEED,
    HORIZON_FILES,
    N_SPLITS,
    TARGET,
    build_fold_assignments,
    company_codes,
    split_indices_for_fold,
)


def load_grouping(root: Path, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the label, country and company columns of one full-panel horizon file.

    The row order is the file's, as ``v4_protocol._load_horizon`` reads it with no cap.

    Args:
        root: Directory holding the parquet files.
        horizon: Paper horizon; see ``HORIZON_FILES`` for the off-by-one file mapping.

    Returns:
        ``(y, country, company)``, one entry per row.
    """
    import pyarrow.parquet as pq

    df = pq.read_table(root / HORIZON_FILES[horizon], columns=[TARGET, "country", "company"]).to_pandas()
    return df[TARGET].to_numpy().astype(np.int64), df["country"].to_numpy(), df["company"].to_numpy()


def reconstruct(
    y: np.ndarray, country: np.ndarray, company: np.ndarray, n_splits: int = N_SPLITS
) -> dict[int, dict[str, np.ndarray]]:
    """Per-fold test and validation labels and company codes, exactly as ``run`` splits them.

    Args:
        y: Label per row.
        country: Country per row.
        company: Company id per row.
        n_splits: Number of folds.

    Returns:
        Fold -> ``y_true``, ``company_test``, ``y_val``, ``company_val``.
    """
    folds = build_fold_assignments(country, company)
    codes = company_codes(company)
    out = {}
    for k in range(n_splits):
        _, va, te = split_indices_for_fold(folds, k, n_splits)
        out[k] = {"y_true": y[te], "company_test": codes[te], "y_val": y[va], "company_val": codes[va]}
    return out


def verify(run: Path, rebuilt: dict[int, dict[str, np.ndarray]]) -> list[int]:
    """Check one run's saved labels against the reconstruction.

    Args:
        run: A `fintfm-v4protocol` output directory.
        rebuilt: :func:`reconstruct`'s output.

    Returns:
        The folds checked.

    Raises:
        SystemExit: If the run used a protocol this cannot rebuild, or any fold differs.
    """
    cfg_file = run / "v4_protocol.json"
    if cfg_file.exists():
        cfg = json.loads(cfg_file.read_text()).get("config", {})
        if cfg.get("max_rows") is not None:
            raise SystemExit(f"{run}: max_rows={cfg['max_rows']}; only full-panel runs are rebuilt")
        if (cfg.get("fold_seed", FOLD_SEED), cfg.get("n_splits", N_SPLITS)) != (FOLD_SEED, N_SPLITS):
            raise SystemExit(f"{run}: fold seed or count differs from the protocol's")
    checked = []
    for f in sorted(run.glob("predictions_fold*.npz")):
        k = int(f.stem.removeprefix("predictions_fold"))
        saved = np.load(f)
        for key in ("y_true", "y_val"):
            if key in saved.files and not np.array_equal(saved[key], rebuilt[k][key]):
                raise SystemExit(f"{f}: saved {key} differs from the reconstruction")
        checked.append(k)
    if not checked:
        raise SystemExit(f"{run}: no predictions_fold*.npz")
    return checked


def main() -> None:
    """Entry point: rebuild, verify against every given run, then write."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, type=Path, help="directory holding the parquet files")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--horizon", type=int, default=0)
    p.add_argument("runs", nargs="+", type=Path, help="run directories to verify against")
    args = p.parse_args()

    y, country, company = load_grouping(args.root, args.horizon)
    rebuilt = reconstruct(y, country, company)
    for run in args.runs:
        print(f"{run}: folds {verify(run, rebuilt)} match the reconstruction")
    args.out.mkdir(parents=True, exist_ok=True)
    for k, arrays in rebuilt.items():
        np.savez_compressed(args.out / f"companies_fold{k}.npz", **arrays)
        n, g = len(arrays["y_true"]), len(np.unique(arrays["company_test"]))
        print(f"fold {k}: {n:,} test rows, {g:,} companies ({n / g:.2f} rows each), "
              f"{int(arrays['y_true'].sum()):,} positive")


if __name__ == "__main__":
    main()
