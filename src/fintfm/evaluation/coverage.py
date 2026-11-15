"""Native coverage: how many tasks a harness attempted, scored and skipped, and why.

Openspec ``adopt-published-methods`` task 36.4. A mean over only the tasks that happened to
score flatters a model, so every harness that scores several tasks, datasets or splits reports
``n_attempted``, ``n_scored`` and ``n_skipped`` with a reason per skip beside its aggregate.
The three counts are tied together on purpose: a task that was attempted and neither scored nor
skipped with a reason is an error, not a quiet drop.
"""

from __future__ import annotations

from typing import Any


class Coverage:
    """Counts of attempted, scored and skipped units of one harness run.

    A unit is whatever the harness aggregates over: a synthetic task, a dataset, a split, a
    (dataset, split, model) row. Every unit is announced with :meth:`attempt` and resolved
    with :meth:`scored` or :meth:`skip`; :meth:`to_dict` refuses to report while one is open.
    """

    def __init__(self) -> None:
        """Start with nothing attempted."""
        self._attempted: list[str] = []
        self._scored: list[str] = []
        self._skipped: dict[str, str] = {}

    def attempt(self, unit: str) -> None:
        """Record that ``unit`` was tried.

        Args:
            unit: A name that is unique within the run.

        Raises:
            ValueError: If ``unit`` was already attempted.
        """
        if unit in self._attempted:
            raise ValueError(f"unit {unit!r} was attempted twice")
        self._attempted.append(unit)

    def scored(self, unit: str) -> None:
        """Record that an attempted ``unit`` produced a score."""
        self._resolve(unit)
        self._scored.append(unit)

    def skip(self, unit: str, reason: str) -> None:
        """Record that an attempted ``unit`` was skipped.

        Args:
            unit: The attempted unit.
            reason: Why it has no score; must not be empty.

        Raises:
            ValueError: If ``reason`` is empty.
        """
        if not reason.strip():
            raise ValueError(f"unit {unit!r} was skipped without a reason")
        self._resolve(unit)
        self._skipped[unit] = reason

    def _resolve(self, unit: str) -> None:
        """Check that ``unit`` was attempted and not yet resolved."""
        if unit not in self._attempted:
            raise ValueError(f"unit {unit!r} was resolved before it was attempted")
        if unit in self._scored or unit in self._skipped:
            raise ValueError(f"unit {unit!r} was resolved twice")

    @property
    def n_attempted(self) -> int:
        """Units tried."""
        return len(self._attempted)

    @property
    def n_scored(self) -> int:
        """Units that produced a score."""
        return len(self._scored)

    @property
    def n_skipped(self) -> int:
        """Units skipped with a reason."""
        return len(self._skipped)

    def to_dict(self) -> dict[str, Any]:
        """The record to put in a harness's JSON.

        Returns:
            ``n_attempted``, ``n_scored``, ``n_skipped``, ``coverage`` (scored over attempted,
            ``None`` when nothing was attempted) and ``skipped`` (unit to reason).

        Raises:
            ValueError: If attempted differs from scored plus skipped, that is a unit was
                attempted and never resolved.
        """
        if self.n_attempted != self.n_scored + self.n_skipped:
            open_units = [u for u in self._attempted
                          if u not in self._scored and u not in self._skipped]
            raise ValueError(
                f"{self.n_attempted} attempted but {self.n_scored} scored + "
                f"{self.n_skipped} skipped; unresolved: {open_units}"
            )
        return {
            "n_attempted": self.n_attempted,
            "n_scored": self.n_scored,
            "n_skipped": self.n_skipped,
            "coverage": self.n_scored / self.n_attempted if self.n_attempted else None,
            "skipped": dict(self._skipped),
        }

    def summary(self, what: str = "tasks") -> str:
        """The printed line, with the skip reasons under it.

        Args:
            what: The unit's plural, for the sentence.

        Returns:
            ``"coverage: 18/20 tasks scored, 2 skipped"`` plus one line per skip.
        """
        rec = self.to_dict()
        head = (f"coverage: {rec['n_scored']}/{rec['n_attempted']} {what} scored, "
                f"{rec['n_skipped']} skipped")
        lines = [head]
        lines += [f"  skipped {u}: {r}" for u, r in rec["skipped"].items()]
        return "\n".join(lines)

    @classmethod
    def from_rows(cls, units: list[str], skipped: dict[str, str]) -> Coverage:
        """Build from a finished table: every unit, and the reason for those without a score.

        Args:
            units: Every attempted unit, in order.
            skipped: Unit to reason, for the units with no score.

        Returns:
            The resolved :class:`Coverage`.
        """
        cov = cls()
        for u in units:
            cov.attempt(u)
            if u in skipped:
                cov.skip(u, skipped[u])
            else:
                cov.scored(u)
        return cov
