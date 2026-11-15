"""Inclusion rule of the forward prediction register as executable code (openspec task 6.3).

The register's cohort must be mechanical so it cannot be curated after the fact. This module
is a pure function from a table of company records to the included company numbers, a
per-company exclusion reason and a deterministic cohort hash (SHA-256) that task 6.4 publishes.
It does no I/O, reads no network and has no clock: the snapshot date is an explicit input.

Input fields (company level only), named as in the Companies House free company data product
(``docs/research/companies_house_source.md``, "Fields in the basic-data snapshot"). Header
spelling in a real file is NOT established (no bulk download was made; the data-fields PDF is
dated 2012), so mapping real headers to these names is the caller's job:

* ``CompanyNumber``, ``CompanyStatus``, ``CompanyCategory``, ``IncorporationDate``,
  ``DissolutionDate``, ``LastMadeUpDate``, ``AccountsCategory``, ``SicText_1`` to ``SicText_4``.

Dates are ``dd/mm/YYYY`` (the snapshot's format, unverified), ISO ``YYYY-MM-DD``, or
``datetime.date``. Only the fields in :data:`READ_COLUMNS` are ever read. Registered office,
"care of", officer and PSC data are never read: the research note recommends keeping personal
data out of the repository and out of the hash inputs, and a test enforces it.

Rules apply in the order of :class:`Reason`; a company gets the first reason that fails.
Every threshold lives in :class:`CohortRule` with its reason. Choices the proposal leaves open
carry the marker ``OWNER DECISION`` and a default that is a placeholder, not an agreed value.
Changing any field of the rule changes :func:`cohort_hash`, so a different rule can never
publish the same hash as another.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any

#: Bump when the logic of :func:`select_cohort` changes (not for threshold changes, which are
#: already part of the hash through :class:`CohortRule`).
RULE_VERSION = "1"

#: The only record fields this module reads. No personal-data field is listed, deliberately.
READ_COLUMNS: tuple[str, ...] = (
    "CompanyNumber",
    "CompanyStatus",
    "CompanyCategory",
    "IncorporationDate",
    "DissolutionDate",
    "LastMadeUpDate",
    "AccountsCategory",
    "SicText_1",
    "SicText_2",
    "SicText_3",
    "SicText_4",
)

_SIC_COLUMNS = READ_COLUMNS[-4:]

INCLUDED = "included"


class Reason:
    """Exclusion reasons, in the order the rules are applied."""

    MALFORMED_NUMBER = "malformed_company_number"
    DUPLICATE_NUMBER = "duplicate_company_number"
    ALREADY_DISSOLVED = "already_dissolved"
    STATUS_NOT_ACTIVE = "status_not_active"
    STRIKE_OFF_PROPOSED = "strike_off_proposed"
    CATEGORY_EXCLUDED = "company_category_excluded"
    TOO_YOUNG = "incorporated_too_recently"
    SIC_MISSING = "sic_code_missing"
    SIC_EXCLUDED = "sic_code_excluded"
    ACCOUNTS_CATEGORY_EXCLUDED = "accounts_category_excluded"
    NO_ACCOUNTS = "no_accounts_filed"
    ACCOUNTS_STALE = "accounts_too_old"


@dataclass(frozen=True)
class CohortRule:
    """Every threshold of the inclusion rule, each with its reason.

    Attributes:
        snapshot_date: Date of the archived monthly snapshot the cohort is drawn from. All
            ages are measured to this date, never to "today", so a rerun is identical.
        allowed_categories: ``CompanyCategory`` values accepted, compared case-insensitively.
            Default private and public limited companies, whose accounts are filed in the
            structured form the model needs. OWNER DECISION: the exact label strings in a real
            snapshot are not established (no file downloaded), and whether to admit
            limited-liability partnerships or other categories is open.
        min_age_months: Minimum months between ``IncorporationDate`` and ``snapshot_date``.
            OWNER DECISION, placeholder 36: the proposal states no minimum accounts history.
            Three years means at least two sets of annual accounts are normally due, which
            lets a model use a lagged year; the snapshot itself holds only the latest
            accounts date, so history cannot be counted from it directly.
        excluded_sic_divisions: Two-digit SIC 2007 divisions excluded. Default 64, 65, 66
            (financial services and insurance), whose balance sheets are not comparable with
            a non-financial credit model, and 98, 99 (undifferentiated household activities,
            extraterritorial bodies). OWNER DECISION: sector scope of the register.
        excluded_sic_codes: Five-digit SIC codes excluded in addition. Default ``99999``
            (dormant company) and ``98000``. OWNER DECISION.
        require_sic: Exclude companies with no usable SIC code (the snapshot uses the text
            "None Supplied"). Default True: sector exclusions cannot be applied to a company
            without a code, so admitting it would make the rule unverifiable.
        excluded_accounts_categories: ``AccountsCategory`` values excluded, compared
            case-insensitively. Default ``NO ACCOUNTS FILED`` and ``DORMANT``: no financial
            statements to predict from. OWNER DECISION: whether micro-entity or other
            categories that file no profit and loss account should be excluded; the label
            strings in a real snapshot are not established.
        max_accounts_age_months: Maximum months between ``LastMadeUpDate`` and
            ``snapshot_date``. OWNER DECISION, placeholder 24: older accounts are stale
            features and suggest a company about to be struck off for non-filing.
        outcome_events: Insolvency case types counted as the outcome ("default"), from the
            Companies House insolvency resource (``cases[].type``). Not used to include or
            exclude, but recorded in the rule so the hash binds the outcome definition.
            OWNER DECISION: the note says only that members' voluntary liquidation is solvent
            and not a default and that strike-off can be dormancy. Placeholder: compulsory
            liquidation, creditors' voluntary liquidation, administration, receivership.
        outcome_horizon_months: Months after the snapshot in which an outcome counts. The
            proposal and task 6.5 say 12 months to the first scoring pass.
    """

    snapshot_date: date
    allowed_categories: tuple[str, ...] = ("Private Limited Company", "Public Limited Company")
    min_age_months: int = 36
    excluded_sic_divisions: tuple[str, ...] = ("64", "65", "66", "98", "99")
    excluded_sic_codes: tuple[str, ...] = ("99999", "98000")
    require_sic: bool = True
    excluded_accounts_categories: tuple[str, ...] = ("NO ACCOUNTS FILED", "DORMANT")
    max_accounts_age_months: int = 24
    outcome_events: tuple[str, ...] = (
        "compulsory-liquidation",
        "creditors-voluntary-liquidation",
        "in-administration",
        "receivership",
    )
    outcome_horizon_months: int = 12

    def canonical(self) -> dict[str, Any]:
        """Return the rule as a JSON-serialisable dict with sorted, normalised sequences.

        Returns:
            A dict where the snapshot date is ISO text and every sequence is sorted and
            lower-cased where comparison is case-insensitive.
        """
        d = asdict(self)
        d["snapshot_date"] = self.snapshot_date.isoformat()
        d["allowed_categories"] = sorted({c.strip().lower() for c in self.allowed_categories})
        d["excluded_accounts_categories"] = sorted(
            {c.strip().lower() for c in self.excluded_accounts_categories}
        )
        for k in ("excluded_sic_divisions", "excluded_sic_codes", "outcome_events"):
            d[k] = sorted(set(d[k]))
        return d


@dataclass(frozen=True)
class CohortResult:
    """Output of :func:`select_cohort`.

    Attributes:
        included: Sorted tuple of included, normalised company numbers.
        reasons: Mapping company number to :data:`INCLUDED` or a :class:`Reason` value, for
            every input record. Malformed numbers are keyed by their raw text.
        cohort_hash: Hex SHA-256 from :func:`cohort_hash`.
        rule_version: :data:`RULE_VERSION` at run time.
    """

    included: tuple[str, ...]
    reasons: dict[str, str]
    cohort_hash: str
    rule_version: str


def normalise_company_number(raw: Any) -> str | None:
    """Normalise a company number to Companies House's 8-character form.

    Args:
        raw: Raw value, e.g. ``"1234567"``, ``" sc123456 "`` or an int.

    Returns:
        Upper-case 8-character alphanumeric number (all-digit numbers zero-padded on the
        left), or ``None`` if the value is empty or not alphanumeric.
    """
    if raw is None:
        return None
    s = str(raw).strip().upper()
    if not s or not s.isalnum() or len(s) > 8:
        return None
    return s.zfill(8) if s.isdigit() else s


def parse_date(raw: Any) -> date | None:
    """Parse a snapshot date.

    Args:
        raw: ``datetime.date``, ``dd/mm/YYYY`` text, ISO ``YYYY-MM-DD`` text, or blank/NaN.

    Returns:
        The date, or ``None`` if blank or unparseable.
    """
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    try:
        if "/" in text:
            day, month, year = text.split("/")
            return date(int(year), int(month), int(day))
        return date.fromisoformat(text)
    except ValueError:
        return None


def _months_between(earlier: date, later: date) -> int:
    """Whole months from ``earlier`` to ``later`` (negative if ``earlier`` is after)."""
    m = (later.year - earlier.year) * 12 + later.month - earlier.month
    return m - 1 if later.day < earlier.day else m


def _sic_codes(rec: Mapping[str, Any]) -> list[str]:
    """Extract five-digit SIC codes from the ``SicText_n`` fields (``"62012 - text"``)."""
    codes = []
    for col in _SIC_COLUMNS:
        v = rec.get(col)
        if isinstance(v, str) and len(v.strip()) >= 5 and v.strip()[:5].isdigit():
            codes.append(v.strip()[:5])
    return codes


def _norm(v: Any) -> str:
    """Case-folded, stripped text of a field; empty for missing values."""
    return v.strip().lower() if isinstance(v, str) else ""


def _decide(rec: Mapping[str, Any], rule: CohortRule) -> str:
    """Return :data:`INCLUDED` or the first failing :class:`Reason` for one record."""
    if parse_date(rec.get("DissolutionDate")) is not None:
        return Reason.ALREADY_DISSOLVED
    status = _norm(rec.get("CompanyStatus"))
    if "proposal to strike off" in status or "proposal-to-strike-off" in status:
        return Reason.STRIKE_OFF_PROPOSED
    if status != "active":
        return Reason.STATUS_NOT_ACTIVE
    allowed = {c.strip().lower() for c in rule.allowed_categories}
    if _norm(rec.get("CompanyCategory")) not in allowed:
        return Reason.CATEGORY_EXCLUDED
    inc = parse_date(rec.get("IncorporationDate"))
    if inc is None or _months_between(inc, rule.snapshot_date) < rule.min_age_months:
        return Reason.TOO_YOUNG
    codes = _sic_codes(rec)
    if not codes and rule.require_sic:
        return Reason.SIC_MISSING
    if any(c in rule.excluded_sic_codes or c[:2] in rule.excluded_sic_divisions for c in codes):
        return Reason.SIC_EXCLUDED
    cat = _norm(rec.get("AccountsCategory"))
    if cat in {c.strip().lower() for c in rule.excluded_accounts_categories}:
        return Reason.ACCOUNTS_CATEGORY_EXCLUDED
    last = parse_date(rec.get("LastMadeUpDate"))
    if last is None:
        return Reason.NO_ACCOUNTS
    if _months_between(last, rule.snapshot_date) > rule.max_accounts_age_months:
        return Reason.ACCOUNTS_STALE
    return INCLUDED


def cohort_hash(included: Iterable[str], rule: CohortRule, rule_version: str = RULE_VERSION) -> str:
    """Deterministic SHA-256 of a cohort and the rule that produced it.

    The payload is the rule version, the canonical JSON of the whole rule (sorted keys) and
    the sorted, de-duplicated company numbers, newline-separated, so the hash does not depend
    on input order and changes with membership, version or any threshold. Recomputable from
    the published number list and rule alone (task 6.4).

    Args:
        included: Normalised company numbers.
        rule: The rule used.
        rule_version: Logic version, default :data:`RULE_VERSION`.

    Returns:
        64-character lower-case hex digest.
    """
    head = json.dumps(
        {"rule_version": rule_version, "rule": rule.canonical()},
        sort_keys=True,
        separators=(",", ":"),
    )
    body = "\n".join(sorted(set(included)))
    return hashlib.sha256(f"{head}\n{body}".encode()).hexdigest()


def select_cohort(records: Iterable[Mapping[str, Any]], rule: CohortRule) -> CohortResult:
    """Apply the inclusion rule to a table of company records.

    Args:
        records: Mappings (e.g. ``DataFrame.to_dict("records")``) carrying the fields of
            :data:`READ_COLUMNS`; other keys are ignored and never read.
        rule: The thresholds, including the snapshot date.

    Returns:
        A :class:`CohortResult`. The result does not depend on record order.
    """
    by_number: dict[str, list[Mapping[str, Any]]] = {}
    reasons: dict[str, str] = {}
    for rec in records:
        raw = rec.get("CompanyNumber")
        num = normalise_company_number(raw)
        if num is None:
            reasons[str(raw)] = Reason.MALFORMED_NUMBER
        else:
            by_number.setdefault(num, []).append(rec)
    for num, recs in by_number.items():
        reasons[num] = Reason.DUPLICATE_NUMBER if len(recs) > 1 else _decide(recs[0], rule)
    included = tuple(sorted(n for n, r in reasons.items() if r == INCLUDED))
    return CohortResult(
        included=included,
        reasons=dict(sorted(reasons.items())),
        cohort_hash=cohort_hash(included, rule),
        rule_version=RULE_VERSION,
    )
