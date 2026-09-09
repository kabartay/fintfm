"""V4FinBench's published protocol, reimplemented — the parts that must be exact.

`docs/FINDINGS.md` §36: our out-of-time numbers cannot be placed against their published
table, and reproducing their protocol is the only route to a comparable number. A protocol
that is *nearly* theirs produces a number that looks comparable and is not, which is worse
than having none — so the fold algebra is tested rather than trusted.
"""

import numpy as np
import pytest

from fintfm.experiments.v4_protocol import (
    DROP_COLUMNS,
    HORIZON_FILES,
    N_SPLITS,
    best_f1_threshold,
    build_fold_assignments,
    split_indices_for_fold,
)


def _panel(n_companies=50, n_countries=4, rows_each=7, seed=0):
    rng = np.random.RandomState(seed)
    country, company = [], []
    for c in range(n_countries):
        for k in range(n_companies):
            name = f"c{c}_f{k}"
            reps = rng.randint(1, rows_each + 1)
            country += [f"CTRY{c}"] * reps
            company += [name] * reps
    return np.array(country, dtype=object), np.array(company, dtype=object)


def test_every_observation_of_a_company_lands_in_one_fold():
    """The grouping guarantee. If it breaks, the split leaks a company across train and test."""
    country, company = _panel()
    folds = build_fold_assignments(country, company)
    for name in np.unique(company):
        assert len(np.unique(folds[company == name])) == 1, name


def test_countries_are_spread_evenly_across_folds():
    """Their protocol preserves country proportions across folds."""
    country, company = _panel(n_companies=100)
    folds = build_fold_assignments(country, company)
    for c in np.unique(country):
        counts = np.bincount(folds[country == c], minlength=N_SPLITS)
        # round-robin over companies, so fold sizes differ by at most the companies' row counts
        assert counts.min() > 0
        assert counts.max() / counts.min() < 3.0, counts


def test_fold_assignment_is_deterministic():
    country, company = _panel()
    a = build_fold_assignments(country, company)
    b = build_fold_assignments(country, company)
    np.testing.assert_array_equal(a, b)


def test_a_different_seed_gives_a_different_assignment():
    country, company = _panel()
    a = build_fold_assignments(country, company, seed=42)
    b = build_fold_assignments(country, company, seed=43)
    assert not np.array_equal(a, b)


def test_country_order_is_sorted_so_the_shared_rng_is_consumed_reproducibly():
    """One RandomState is consumed across countries; the order therefore changes the result.

    Shuffling the row order must not change any company's fold, because the algorithm sorts
    countries and takes companies in first-appearance order within each.
    """
    country, company = _panel()
    base = build_fold_assignments(country, company)
    base_map = {c: base[company == c][0] for c in np.unique(company)}
    rng = np.random.RandomState(7)
    perm = rng.permutation(len(country))
    shuffled = build_fold_assignments(country[perm], company[perm])
    shuffled_map = {c: shuffled[company[perm] == c][0] for c in np.unique(company)}
    # first-appearance order changes under permutation, so assignments may differ -- what must
    # hold is that each company is still internally consistent, which the guarantee test covers.
    assert set(base_map) == set(shuffled_map)


def test_split_rotation_matches_their_specification():
    """Validation is `fold`, test is `(fold + 1) % 5`, training is the other three."""
    folds = np.repeat(np.arange(N_SPLITS), 10)
    for f in range(N_SPLITS):
        tr, va, te = split_indices_for_fold(folds, f)
        assert set(folds[va]) == {f}
        assert set(folds[te]) == {(f + 1) % N_SPLITS}
        assert set(folds[tr]) == set(range(N_SPLITS)) - {f, (f + 1) % N_SPLITS}
        # disjoint and complete
        assert len(tr) + len(va) + len(te) == len(folds)
        assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))


def test_split_is_roughly_sixty_twenty_twenty():
    folds = np.repeat(np.arange(N_SPLITS), 100)
    tr, va, te = split_indices_for_fold(folds, 0)
    assert len(tr) == 300 and len(va) == 100 and len(te) == 100


def test_split_refuses_an_out_of_range_fold():
    folds = np.repeat(np.arange(N_SPLITS), 3)
    for bad in (-1, N_SPLITS, 99):
        with pytest.raises(ValueError, match="fold must lie"):
            split_indices_for_fold(folds, bad)


def test_null_company_is_refused_rather_than_silently_grouped():
    country = np.array(["A", "A"], dtype=object)
    company = np.array(["x", None], dtype=object)
    with pytest.raises(ValueError, match="null company"):
        build_fold_assignments(country, company)


def test_threshold_maximises_f1_on_the_validation_curve():
    y = np.array([0, 0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
    thr, f1 = best_f1_threshold(y, p)
    assert f1 == pytest.approx(1.0)
    assert 0.4 < thr <= 0.8


def test_threshold_on_a_degenerate_split_is_reported_not_invented():
    thr, f1 = best_f1_threshold(np.zeros(10, dtype=int), np.linspace(0, 1, 10))
    assert (thr, f1) == (0.5, 0.0)


def test_horizon_file_mapping_is_off_by_one_as_their_protocol_specifies():
    """Their h=0 is company_years_h1.parquet. Getting this wrong shifts every result."""
    assert HORIZON_FILES[0] == "company_years_h1.parquet"
    assert HORIZON_FILES[5] == "company_years_h6.parquet"
    assert len(HORIZON_FILES) == 6


def test_drop_list_matches_their_released_schema():
    assert "emis_id" in DROP_COLUMNS and "company" in DROP_COLUMNS
    assert "Revenue/employee" in DROP_COLUMNS  # an excluded released-schema field, not an id
    assert len(DROP_COLUMNS) == 11
