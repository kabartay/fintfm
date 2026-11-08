"""The time-based splitter refuses leakage across the boundary."""

import numpy as np
import pytest

from fintfm.evaluation.splits import time_based_split


def _panel():
    year = np.array([2010, 2011, 2012, 2017, 2018, 2019])
    firm = np.array(["a", "b", "c", "d", "e", "f"])
    return year, firm


def test_clean_panel_splits_by_year():
    year, firm = _panel()
    s = time_based_split(year, firm, 2016, 2017)
    assert s.train.tolist() == [0, 1, 2] and s.test.tolist() == [3, 4, 5]
    assert s.n_shared_ids == 0 and s.n_train_dropped == 0


def test_overlapping_panel_raises():
    year, firm = _panel()
    firm[4] = "a"  # firm a is observed in 2010 and again in 2018
    with pytest.raises(ValueError, match="both windows"):
        time_based_split(year, firm, 2016, 2017)


def test_overlap_is_dropped_from_train_only_when_asked():
    year, firm = _panel()
    firm[4] = "a"
    s = time_based_split(year, firm, 2016, 2017, on_overlap="drop_from_train")
    assert s.train.tolist() == [1, 2] and s.test.tolist() == [3, 4, 5]
    assert (s.n_shared_ids, s.n_train_dropped) == (1, 1)
    assert not set(firm[s.train]) & set(firm[s.test])


def test_overlapping_windows_and_bad_input_raise():
    year, firm = _panel()
    with pytest.raises(ValueError, match="windows overlap"):
        time_based_split(year, firm, 2017, 2017)
    with pytest.raises(ValueError, match="empty window"):
        time_based_split(year, firm, 2005, 2006)
    with pytest.raises(ValueError, match="equal 1-D"):
        time_based_split(year, firm[:3], 2016, 2017)
    bad = year.astype(float)
    bad[0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        time_based_split(bad, firm, 2016, 2017)
