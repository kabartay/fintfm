"""Tests for the dependency-free SVG line-plot writer."""

import xml.etree.ElementTree as ET

import pytest

from fintfm.evaluation.svgplot import line_plot

NS = "{http://www.w3.org/2000/svg}"


def _plot(**kw):
    series = {"a": [(1, 0.5), (10, 0.7), (100, 0.9)], "b": [(1, 0.4), (100, 0.6)]}
    return line_plot(series, title="t", x_label="x", y_label="y", **kw)


def test_parses_with_one_polyline_per_series_and_a_white_background():
    root = ET.fromstring(_plot(reference_lines={"chance": 0.5}))
    assert len(root.findall(f"{NS}polyline")) == 2
    assert len(root.findall(f"{NS}circle")) == 5
    assert root.find(f"{NS}rect").get("fill") == "#ffffff"
    texts = [t.text for t in root.iter(f"{NS}text")]
    assert "a" in texts and "b" in texts and "chance" in texts


def test_deterministic():
    assert _plot(log_x=True) == _plot(log_x=True)


def test_log_x_mapping_is_monotone_and_equal_per_decade():
    root = ET.fromstring(_plot(log_x=True))
    pl = next(p for p in root.findall(f"{NS}polyline") if p.get("data-series") == "a")
    xs = [float(p.split(",")[0]) for p in pl.get("points").split()]
    assert xs == sorted(xs) and len(set(xs)) == 3
    assert xs[1] - xs[0] == pytest.approx(xs[2] - xs[1], abs=0.2)


def test_escaping():
    svg = line_plot({"a<b&c": [(0, 0), (1, 1)]}, title="<x>&", x_label='"q"', y_label="y")
    root = ET.fromstring(svg)
    assert "a<b&c" in [t.text for t in root.iter(f"{NS}text")]


def test_bad_input_refused():
    with pytest.raises(ValueError):
        line_plot({"a": []}, title="t", x_label="x", y_label="y")
    with pytest.raises(ValueError):
        line_plot({"a": [(0, 1)]}, title="t", x_label="x", y_label="y", log_x=True)
