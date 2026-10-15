"""Paired per-fold comparison of `fintfm-v4protocol` outputs against a control, §144-style.

Reproduces the table format §144-§147 used -- control AP, arm AP, paired-bootstrap difference
and Holm-adjusted p per fold, Holm across that arm's folds -- so a re-score (§158's re-adjudication)
is judged exactly the way the original results were. Refuses to compare a fold whose rows differ
between control and arm, since a difference on different rows means nothing.

Usage:
    uv run python scripts/compare_v4_scores.py \\
        --control runs/task48-6-kaggle/control_rescore.json \\
        s1=runs/cellattn-s1-kaggle/s1_rescore.json

``--cluster`` adds a company-level bootstrap beside the row bootstrap (audit AU2.1): a test fold
holds about five rows per company, so the row interval is too narrow. Runs saved before the npz
carried ``company_test`` need ``--companies`` pointing at ``scripts/v4_fold_companies.py``'s
output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from fintfm.evaluation.metrics import holm_adjusted_p, paired_auc_difference


def _folds(run: Path) -> dict[int, dict[str, np.ndarray]]:
    """Load every `predictions_fold<k>.npz` in a v4protocol output directory.

    Args:
        run: The directory `fintfm-v4protocol --out` wrote.

    Returns:
        Fold index -> arrays (`y_true`, `pred_fintfm`, ...).
    """
    out = {}
    for f in sorted(run.glob("predictions_fold*.npz")):
        d = np.load(f)
        out[int(f.stem.removeprefix("predictions_fold"))] = {k: d[k] for k in d.files}
    return out


#: Settings that decide *which rows* and *which protocol* a run scored. Two runs that differ in
#: any of these are not a paired comparison even when their `y_true` arrays happen to match;
#: inference and fine-tuning settings are what an arm varies, so they are not checked.
PROTOCOL_KEYS = ("horizon", "max_rows", "n_features", "fold_seed", "n_splits", "dataset")


def check_protocol(control: Path, arm: Path) -> None:
    """Refuse a comparison whose two runs used different protocol settings.

    Reads each directory's `v4_protocol.json`. A directory without one (an older or partial
    run) is reported and passed, since its rows are still checked fold by fold.

    Args:
        control: The control's `fintfm-v4protocol` output directory.
        arm: The arm's.

    Raises:
        SystemExit: If any :data:`PROTOCOL_KEYS` setting differs.
    """
    recs = []
    for d in (control, arm):
        f = d / "v4_protocol.json"
        if not f.exists():
            print(f"note: {d} has no v4_protocol.json; protocol settings not checked")
            return
        rec = json.loads(f.read_text())
        recs.append({**rec.get("config", {}), "dataset": rec.get("dataset")})
    diff = {k: (recs[0].get(k), recs[1].get(k)) for k in PROTOCOL_KEYS
            if recs[0].get(k) != recs[1].get(k)}
    if diff:
        raise SystemExit(f"{arm} vs {control}: protocol settings differ {diff} -- not comparable")


def fold_companies(
    k: int, y: np.ndarray, ctrl: dict[str, np.ndarray], arm: dict[str, np.ndarray],
    companies: Path | None,
) -> np.ndarray:
    """The company of every test row of fold ``k``, for a cluster bootstrap.

    Runs since AU2.1 save ``company_test`` in their npz; older runs do not, and their companies
    come from ``scripts/v4_fold_companies.py``'s reconstruction instead.

    Args:
        k: Fold index.
        y: The fold's test labels, already checked equal between control and arm.
        ctrl: The control's arrays for this fold.
        arm: The arm's.
        companies: Directory of ``companies_fold<k>.npz`` from ``v4_fold_companies.py``, or
            ``None``.

    Returns:
        ``(n,)`` company codes aligned with ``y``.

    Raises:
        SystemExit: If no source has them, or two sources disagree.
    """
    saved = [d["company_test"] for d in (ctrl, arm) if "company_test" in d]
    if len(saved) == 2 and not np.array_equal(*saved):
        raise SystemExit(f"fold {k}: control and arm record different companies -- not comparable")
    if saved:
        return saved[0]
    if companies is None:
        raise SystemExit(f"fold {k}: no company_test in either run; pass --companies "
                         "(written by scripts/v4_fold_companies.py)")
    rec = np.load(companies / f"companies_fold{k}.npz")
    if not np.array_equal(rec["y_true"], y):
        raise SystemExit(f"fold {k}: reconstructed companies do not match this fold's rows")
    return rec["company_test"]


def compare(
    control: Path, arm_dir: Path, n_boot: int = 2000, cluster: bool = False,
    companies: Path | None = None,
) -> list[dict]:
    """Per-fold paired AP comparison of one arm against the control.

    Args:
        control: The control's `fintfm-v4protocol` output directory.
        arm_dir: The arm's.
        n_boot: Bootstrap resamples per fold.
        cluster: Also bootstrap whole companies (audit AU2.1), beside the row bootstrap.
        companies: Reconstructed companies for runs that predate saving them; see
            :func:`fold_companies`.

    Returns:
        One dict per fold: ``fold``, ``control``, ``arm``, ``diff``, ``ci``, ``p``, ``holm``, and
        with ``cluster`` also ``ci_company``, ``p_company``, ``holm_company``. Holm runs across
        this arm's folds, separately for each bootstrap.

    Raises:
        SystemExit: If the runs share no folds, or a fold's rows or protocol differ.
    """
    check_protocol(control, arm_dir)
    ctrl, arm = _folds(control), _folds(arm_dir)
    folds = sorted(set(ctrl) & set(arm))
    if not folds:
        raise SystemExit(f"{arm_dir}: no folds in common with the control")
    rows = []
    for k in folds:
        y = ctrl[k]["y_true"]
        if not np.array_equal(y, arm[k]["y_true"]):
            raise SystemExit(f"{arm_dir}: fold {k} rows differ from the control's -- not comparable")
        a, c = arm[k]["pred_fintfm"], ctrl[k]["pred_fintfm"]
        d, ci, pv = paired_auc_difference(y, a, c, n_boot=n_boot, metric=average_precision_score)
        row = {"fold": k, "control": average_precision_score(y, c),
               "arm": average_precision_score(y, a), "diff": d, "ci": ci, "p": pv}
        if cluster:
            g = fold_companies(k, y, ctrl[k], arm[k], companies)
            _, row["ci_company"], row["p_company"] = paired_auc_difference(
                y, a, c, n_boot=n_boot, metric=average_precision_score, groups=g
            )
        rows.append(row)
    for suffix in ("", "_company") if cluster else ("",):
        for row, h in zip(rows, holm_adjusted_p([r["p" + suffix] for r in rows]), strict=True):
            row["holm" + suffix] = h
    return rows


def main() -> None:
    """Entry point: print one §144-style table per arm."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control", required=True, type=Path)
    p.add_argument("arms", nargs="+", help="name=directory")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--cluster", action="store_true",
                   help="also resample whole companies, reported beside the row bootstrap")
    p.add_argument("--companies", type=Path, default=None,
                   help="companies_fold<k>.npz directory from scripts/v4_fold_companies.py, "
                        "for runs saved before predictions carried company_test")
    args = p.parse_args()
    for spec in args.arms:
        name, _, path = spec.partition("=")
        rows = compare(args.control, Path(path), args.n_boot, args.cluster, args.companies)
        print(f"\n{name} vs control ({args.control})")
        if args.cluster:
            print("| fold | control AP | arm AP | diff | 95% CI (rows) | Holm p (rows) "
                  "| 95% CI (companies) | Holm p (companies) |")
            print("| --- | --- | --- | --- | --- | --- | --- | --- |")
        else:
            print("| fold | control AP | arm AP | diff | 95% CI | Holm-adjusted p |")
            print("| --- | --- | --- | --- | --- | --- |")
        for r in rows:
            (lo, hi) = r["ci"]
            line = (f"| {r['fold']} | {r['control']:.4f} | {r['arm']:.4f} | {r['diff']:+.4f} "
                    f"| [{lo:+.4f}, {hi:+.4f}] | {r['holm']:.3f} |")
            if args.cluster:
                (clo, chi) = r["ci_company"]
                line += f" [{clo:+.4f}, {chi:+.4f}] | {r['holm_company']:.3f} |"
            print(line)
        diffs = [r["diff"] for r in rows]
        print(f"mean diff {float(np.mean(diffs)):+.4f} over {len(rows)} fold(s); "
              f"{sum(d < 0 for d in diffs)} negative, {sum(d > 0 for d in diffs)} positive")


if __name__ == "__main__":
    main()
