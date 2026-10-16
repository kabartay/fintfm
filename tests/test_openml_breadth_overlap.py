"""The OpenML breadth suite must not overlap the benchmarks it is meant to steer towards.

`fintfm.experiments.openml_breadth.SUITE` is a decision loop: arms are chosen on it. Six of its
original fifteen datasets were TabArena-v0.1 datasets (design review 2026-10-16, task DR4.1),
so choosing on it biased the eventual TabArena number. This is evaluation, not training, so
decision D2 is untouched; the harm is selection bias, and nothing would warn of it.

The TabArena-v0.1 and BeyondArena lists live in `fintfm.evaluation.overlap`, which records
where they were copied from and what the checks do not cover (a row-hash comparison). The
proxy Elo suite (task DR4.2) is held to the same lists, so there is one copy to keep current.
"""

from __future__ import annotations

from fintfm.evaluation.overlap import BEYONDARENA, TABARENA_V0_1, same_dataset_name
from fintfm.experiments.openml_breadth import SUITE

#: The OpenML dataset id each SUITE entry resolves to at its pinned version, read from the local
#: scikit-learn OpenML cache (``~/scikit_learn_data/openml``) on 2026-10-16. A new SUITE entry
#: must add its id here, which the first test enforces.
SUITE_OPENML_IDS: dict[str, int] = {
    "mammography": 310, "wilt": 1570, "ozone-level-8hr": 1487, "pc4": 1049, "kc1": 1067,
    "MagicTelescope": 1120, "phoneme": 1489, "spambase": 44, "banknote-authentication": 1462,
}


def test_the_embedded_lists_are_complete():
    """Guards the constants themselves, so a truncated copy cannot make the overlap test pass."""
    assert len(TABARENA_V0_1) == 51
    assert len({i for i, _ in TABARENA_V0_1}) == 51
    assert len(set(BEYONDARENA)) == 142
    assert set(SUITE_OPENML_IDS) == {name for name, _ in SUITE}


def test_suite_shares_no_openml_id_or_name_with_tabarena_or_beyondarena():
    """Task DR4.1: no SUITE dataset is a TabArena-v0.1 or BeyondArena dataset."""
    tabarena_ids = {i for i, _ in TABARENA_V0_1}
    tabarena_names = [n for _, n in TABARENA_V0_1]
    clashes = []
    for name, _version in SUITE:
        if SUITE_OPENML_IDS.get(name) in tabarena_ids:
            clashes.append((name, "TabArena-v0.1 id"))
        clashes += [
            (name, f"TabArena-v0.1 {t}") for t in tabarena_names if same_dataset_name(name, t)
        ]
        clashes += [(name, f"BeyondArena {b}") for b in BEYONDARENA if same_dataset_name(name, b)]
    assert clashes == []
