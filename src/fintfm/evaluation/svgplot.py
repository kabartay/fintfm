"""Dependency-free SVG line plots for the experiment harnesses.

matplotlib is not a dependency of this project, yet several openspec Verify clauses ask for a
plotted curve. :func:`line_plot` writes a standalone SVG document using only the standard
library: axes, ticks with labels, a legend, one polyline with markers per series and optional
horizontal reference lines (chance level, a Bayes ceiling). The output is deterministic (same
input, same bytes), has an explicit white background so it reads in light and dark viewers, and
escapes all text.
"""

from __future__ import annotations

import math
from xml.sax.saxutils import escape, quoteattr

#: Series colours (colour-blind-safe Okabe-Ito subset), cycled in series order.
PALETTE: tuple[str, ...] = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
    "#F0E442",
)

_W, _H = 720, 440
_ML, _MR, _MT, _MB = 70, 190, 44, 56  # plot-area margins; the right one holds the legend


def _nice_ticks(lo: float, hi: float, target: int = 6) -> list[float]:
    """Return round tick positions covering ``[lo, hi]`` (a 1/2/5 x 10^k step)."""
    if hi <= lo:
        hi = lo + 1.0
    raw = (hi - lo) / max(target, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 5, 10) if m * mag >= raw)
    first = math.ceil(lo / step - 1e-9)
    last = math.floor(hi / step + 1e-9)
    return [round(i * step, 12) for i in range(first, last + 1)]


def _fmt(v: float) -> str:
    """Format a tick or value label compactly and deterministically."""
    return f"{v:.6g}"


def line_plot(
    series: dict[str, list[tuple[float, float]]],
    *,
    title: str,
    x_label: str,
    y_label: str,
    log_x: bool = False,
    y_range: tuple[float, float] | None = None,
    reference_lines: dict[str, float] | None = None,
) -> str:
    """Render line series as a standalone SVG document.

    Args:
        series: Legend label -> ``(x, y)`` points; each series is drawn in the given order
            (sorted by x), one polyline plus one marker per point. NaN points are skipped.
        title: Plot title.
        x_label: X axis label.
        y_label: Y axis label.
        log_x: Place x on a base-10 logarithmic scale (all x must be positive).
        y_range: Fixed ``(low, high)`` y limits; default is the data (and reference) extent
            with a small margin.
        reference_lines: Legend label -> y value of a dashed horizontal line.

    Returns:
        The SVG text.

    Raises:
        ValueError: If there is no finite point, or ``log_x`` meets a non-positive x.
    """
    refs = reference_lines or {}
    pts = {
        k: sorted((float(x), float(y)) for x, y in v if math.isfinite(x) and math.isfinite(y))
        for k, v in series.items()
    }
    xs = [x for v in pts.values() for x, _ in v]
    ys = [y for v in pts.values() for _, y in v]
    if not xs:
        raise ValueError("line_plot needs at least one finite point")
    if log_x and min(xs) <= 0:
        raise ValueError("log_x requires positive x values")

    def tx(x: float) -> float:
        """Map an x value to the axis scale (identity or log10)."""
        return math.log10(x) if log_x else x

    x_lo, x_hi = tx(min(xs)), tx(max(xs))
    if x_hi == x_lo:
        x_lo, x_hi = x_lo - 0.5, x_hi + 0.5
    if y_range is not None:
        y_lo, y_hi = y_range
    else:
        allv = ys + [float(v) for v in refs.values()]
        y_lo, y_hi = min(allv), max(allv)
        pad = (y_hi - y_lo) * 0.05 or 0.5
        y_lo, y_hi = y_lo - pad, y_hi + pad
    pw, ph = _W - _ML - _MR, _H - _MT - _MB

    def px(x: float) -> float:
        """Pixel x of a data x."""
        return _ML + (tx(x) - x_lo) / (x_hi - x_lo) * pw

    def py(y: float) -> float:
        """Pixel y of a data y (SVG y grows downwards)."""
        return _MT + (1.0 - (y - y_lo) / (y_hi - y_lo)) * ph

    out = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{_W}" height="{_H}" '
            f'viewBox="0 0 {_W} {_H}" font-family="sans-serif" font-size="12">'
        ),
        f"<title>{escape(title)}</title>",
        f'<rect width="{_W}" height="{_H}" fill="#ffffff"/>',
        (
            f'<text x="{_ML + pw / 2:.1f}" y="24" text-anchor="middle" font-size="15" '
            f'fill="#111111">{escape(title)}</text>'
        ),
    ]
    # Ticks and grid.
    if log_x:
        xticks = [float(10**e) for e in range(math.ceil(x_lo - 1e-9), math.floor(x_hi + 1e-9) + 1)]
        if len(xticks) < 2:
            xticks = sorted(set(xs))
    else:
        xticks = _nice_ticks(x_lo, x_hi)
    for t in xticks:
        x = px(t)
        out.append(f'<line x1="{x:.1f}" y1="{_MT}" x2="{x:.1f}" y2="{_MT + ph}" stroke="#e5e5e5"/>')
        out.append(
            f'<text x="{x:.1f}" y="{_MT + ph + 16}" text-anchor="middle" fill="#333333">'
            f"{escape(_fmt(t))}</text>"
        )
    for t in _nice_ticks(y_lo, y_hi):
        y = py(t)
        out.append(f'<line x1="{_ML}" y1="{y:.1f}" x2="{_ML + pw}" y2="{y:.1f}" stroke="#e5e5e5"/>')
        out.append(
            f'<text x="{_ML - 8}" y="{y + 4:.1f}" text-anchor="end" fill="#333333">'
            f"{escape(_fmt(t))}</text>"
        )
    out.append(
        f'<rect x="{_ML}" y="{_MT}" width="{pw}" height="{ph}" fill="none" stroke="#333333"/>'
    )
    out.append(
        f'<text x="{_ML + pw / 2:.1f}" y="{_H - 12}" text-anchor="middle" fill="#111111">'
        f"{escape(x_label)}</text>"
    )
    cy = _MT + ph / 2
    out.append(
        f'<text x="16" y="{cy:.1f}" text-anchor="middle" fill="#111111" '
        f'transform="rotate(-90 16 {cy:.1f})">{escape(y_label)}</text>'
    )
    # Reference lines, then series.
    legend: list[tuple[str, str, bool]] = []
    for label, val in refs.items():
        y = py(float(val))
        out.append(
            f'<line x1="{_ML}" y1="{y:.1f}" x2="{_ML + pw}" y2="{y:.1f}" stroke="#777777" '
            f'stroke-dasharray="6 4"><title>{escape(label)}</title></line>'
        )
        legend.append((label, "#777777", True))
    for i, (label, v) in enumerate(pts.items()):
        colour = PALETTE[i % len(PALETTE)]
        legend.append((label, colour, False))
        if not v:
            continue
        coords = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in v)
        out.append(
            f'<polyline points="{coords}" fill="none" stroke="{colour}" stroke-width="2" '
            f"data-series={quoteattr(label)}/>"
        )
        for x, y in v:
            out.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3.5" fill="{colour}"/>')
    # Legend.
    lx, ly = _ML + pw + 16, _MT + 8
    for j, (label, colour, dashed) in enumerate(legend):
        y = ly + j * 18
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        out.append(
            f'<line x1="{lx}" y1="{y}" x2="{lx + 24}" y2="{y}" stroke="{colour}" '
            f'stroke-width="2"{dash}/>'
        )
        out.append(f'<text x="{lx + 30}" y="{y + 4}" fill="#111111">{escape(label)}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"
