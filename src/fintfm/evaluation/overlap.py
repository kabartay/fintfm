"""The datasets of TabArena-v0.1 and BeyondArena, for keeping other suites disjoint from them.

Two suites here steer decisions: the OpenML breadth suite (`fintfm.experiments.openml_breadth`)
and the proxy Elo suite (`fintfm.evaluation.proxy_suite`). Arms chosen on a suite that shares
datasets with TabArena bias the eventual TabArena number (design review 2026-10-16, tasks DR4.1
and DR4.2). This is evaluation, not training, so decision D2 is untouched; the harm is
selection bias, and nothing would warn of it.

The lists below are copied from the local TabArena fork (``~/proj/tabarena``, branch
``fintfm-audit-fixes`` at commit 2d8f7cfba072379587a78a820ec079d0e51fe3fa), so the checks run
without the fork or a network:

* TabArena-v0.1: ``packages/tabarena/src/tabarena/benchmark/task/metadata/data/
  curated_tabarena_dataset_metadata.csv``, columns ``dataset_id`` and ``dataset_name`` (51
  datasets; the same names as ``sources/data/TabArena-v0.1_tasks_metadata.csv``).
* BeyondArena: ``packages/tabarena/src/tabarena/benchmark/task/metadata/sources/data/
  BeyondArena_tasks_metadata.csv``, columns ``dataset_name``, ``target_name`` and
  ``data_foundry_uri`` (142 datasets, one row per dataset of its 3,722 task rows; no OpenML
  ids). Copied to ``beyondarena_datasets.csv`` beside this module, read into
  :data:`BEYONDARENA_DATASETS`. Its ``parquet_bytes`` column is not from the fork: it is each
  file's size in the Hugging Face listing of :data:`BEYONDARENA_REPO` at
  :data:`BEYONDARENA_REVISION`, read on :data:`BEYONDARENA_LISTED` without downloading.

**BeyondArena's files.** The data lives in the Hugging Face dataset repo
:data:`BEYONDARENA_REPO` (873 files: 158 parquet, the rest JSON metadata and a few top-level
files). Each ``data_foundry_uri`` is a container folder, ``<name>/<uuid>`` or
``<name>/versions/<uuid>``, and in the listing at :data:`BEYONDARENA_REVISION` every one of the
142 holds exactly one ``dataset.parquet``, which is the dataset (9.74 GB in all; the largest,
``maps_router_eta``, is 3.5 GB, 1,250,000 rows by 989 columns). The other 16 parquet files are
``tabarena_text_cache_*.parquet`` text embeddings, never read here. Beside each dataset sits
``task_metadata.predictive-ml-task-mold-v1.json``, whose ``target_column_name``
:func:`beyondarena_parquet` checks against the fork's ``target_name`` before the target is
dropped, so a disagreement is reported rather than hashed as a feature.

TabArena's ids are its curated re-uploads (46904 onwards), not the original OpenML uploads other
suites fetch (bank-marketing is 1461 on OpenML-CC18 and 46910 in TabArena), so an id check alone
passes almost anything; the name check is the one that catches overlaps. A dataset re-uploaded
under an unrelated name (the two halves of ``wine_quality`` as ``white_wine`` and ``red_wine``)
passes both, which is why a suite built on these checks also keeps its own list of same-source
exclusions.

**Contents (DR4.1, DR4.2).** :func:`content_fingerprint` hashes a dataset's rows and columns so
a re-upload under any name is caught by what it holds. TabArena's re-uploads rename columns,
reorder rows and rewrite category labels (credit-g's ``<0`` is ``<0 DM`` in 46918), so a hash
of the raw rows would miss them. Two signals survive all three:

* *row hashes*: each row's numeric feature values, rounded to :data:`SIGNIFICANT` significant
  figures and **sorted within the row** (so column order does not matter), hashed; category
  cells and the target are left out on both sides. The share of a candidate's hashed rows
  found among a reference's is the row share.
* *column fingerprints*: each numeric feature with at least :data:`MIN_DISTINCT_NUMERIC`
  distinct values hashed as its sorted rounded values, and each categorical feature with at
  least :data:`MIN_DISTINCT_CATEGORIES` levels hashed as its sorted level counts, which no
  relabelling changes. The share of a candidate's fingerprinted columns found in a reference
  is the column share.

A candidate overlaps a reference when either share reaches :data:`OVERLAP_SHARE` (columns:
with at least :data:`MIN_SHARED_COLUMNS` shared). A dataset subsampled or with its numeric
columns transformed on re-upload passes both signals.

**References.** Candidates are fingerprinted in memory by :func:`content_fingerprint`; every
reference goes through :func:`parquet_fingerprint`, which gives the same result from a parquet
file read a column and then a batch of rows at a time, and can keep only the row hashes some
candidate has, so a 3.5 GB reference never sits in memory. TabArena-v0.1's datasets are read
from openml-python's cache (:data:`TABARENA_OPENML_CACHE`), where the fork's runs left 27 of
the 51; :func:`fetch_tabarena` adds the rest through openml-python, and nothing else here writes
to that cache. BeyondArena's parquet files come from the Hugging Face cache, filled by
:func:`beyondarena_parquet` with ``fetch=True``. Two readers use these references:

* the ``--row-hash`` checks of the two suites' command lines
  (:func:`load_reference_fingerprints`), which compare against whichever TabArena-v0.1
  datasets are already cached, BeyondArena not at all, and say what they lacked;
* the full report (``python -m fintfm.evaluation.overlap --report``, module
  :mod:`fintfm.evaluation.overlap_report`), which compares every suite dataset with all 51
  TabArena-v0.1 and all 142 BeyondArena datasets and writes the result to a JSON file that
  ``tests/test_overlap_report.py`` checks without a network.

**Selection (DR4.1).** The Polish and Taiwanese bankruptcy panels are TabArena-v0.1 datasets
(``polish_companies_bankruptcy``, ``taiwanese_bankruptcy_prediction``) as well as this
project's own credit panels. They may be reported on, never used to choose an arm:
:data:`SELECTION_FORBIDDEN` and :func:`guard_selection` are the one rule every arm-selection
tool calls, overridden only by :data:`REPORTING_FLAG`.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd

#: TabArena-v0.1's 51 datasets as (OpenML dataset id, name). Source in the module docstring.
TABARENA_V0_1: tuple[tuple[int, str], ...] = (
    (46904, "airfoil_self_noise"), (46905, "Amazon_employee_access"), (46906, "anneal"),
    (46907, "Another-Dataset-on-used-Fiat-500"), (46908, "APSFailure"),
    (46910, "bank-marketing"), (46911, "Bank_Customer_Churn"), (46912, "Bioresponse"),
    (46913, "blood-transfusion-service-center"), (46915, "churn"),
    (46916, "coil2000_insurance_policies"), (46917, "concrete_compressive_strength"),
    (46918, "credit-g"), (46919, "credit_card_clients_default"),
    (46920, "customer_satisfaction_in_airline"), (46921, "diabetes"), (46922, "Diabetes130US"),
    (46923, "diamonds"), (46924, "E-CommereShippingData"), (46927, "Fitness_Club"),
    (46928, "Food_Delivery_Time"), (46929, "GiveMeSomeCredit"),
    (46930, "hazelnut-spread-contaminant-detection"), (46931, "healthcare_insurance_expenses"),
    (46932, "heloc"), (46933, "hiva_agnostic"), (46934, "houses"),
    (46935, "HR_Analytics_Job_Change_of_Data_Scientists"),
    (46937, "in_vehicle_coupon_recommendation"), (46938, "Is-this-a-good-customer"),
    (46939, "kddcup09_appetency"), (46940, "Marketing_Campaign"),
    (46941, "maternal_health_risk"), (46942, "miami_housing"),
    (46947, "online_shoppers_intention"), (46949, "physiochemical_protein"),
    (46950, "polish_companies_bankruptcy"), (46952, "qsar-biodeg"), (46953, "QSAR-TID-11"),
    (46954, "QSAR_fish_toxicity"), (46955, "SDSS17"), (46956, "seismic-bumps"),
    (46958, "splice"), (46960, "students_dropout_and_academic_success"),
    (46961, "superconductivity"), (46962, "taiwanese_bankruptcy_prediction"),
    (46963, "website_phishing"), (46964, "wine_quality"), (46969, "NATICUSdroid"),
    (46979, "jm1"), (46980, "MIC"),
)

#: Where BeyondArena's files live: a Hugging Face *dataset* repository.
BEYONDARENA_REPO = "TabArena/BeyondArena"

#: The commit of :data:`BEYONDARENA_REPO` every download is pinned to, so a later revision of
#: the repository cannot change what the report compared against.
BEYONDARENA_REVISION = "2ecfe882ccfb814fc27c4de10a64ceefd5d7655c"

#: Date the file listing of :data:`BEYONDARENA_REPO` at :data:`BEYONDARENA_REVISION` was read.
BEYONDARENA_LISTED = "2026-10-19"


@dataclass(frozen=True)
class BeyondArenaDataset:
    """One BeyondArena dataset, as the fork's metadata and the Hugging Face listing give it.

    Attributes:
        name: The fork's ``dataset_name``, e.g. ``"credit_g"``.
        target: The fork's ``target_name``, the column dropped before hashing.
        uri: The fork's ``data_foundry_uri``: the container folder in
            :data:`BEYONDARENA_REPO`, ``<name>/<uuid>`` or ``<name>/versions/<uuid>``.
        parquet_bytes: Size of :attr:`parquet_file` in the listing at
            :data:`BEYONDARENA_REVISION`.
    """

    name: str
    target: str
    uri: str
    parquet_bytes: int

    @property
    def parquet_file(self) -> str:
        """The dataset's parquet file within the repository."""
        return f"{self.uri}/dataset.parquet"

    @property
    def task_metadata_file(self) -> str:
        """The container's task metadata, which names the target column."""
        return f"{self.uri}/task_metadata.predictive-ml-task-mold-v1.json"

    @property
    def label(self) -> str:
        """The dataset's label in reports, e.g. ``"BeyondArena credit_g"``."""
        return f"BeyondArena {self.name}"


def _read_beyondarena() -> tuple[BeyondArenaDataset, ...]:
    """The rows of ``beyondarena_datasets.csv``, in its order (sorted by name)."""
    text = (files("fintfm.evaluation") / "beyondarena_datasets.csv").read_text(encoding="utf-8")
    return tuple(
        BeyondArenaDataset(r["dataset_name"], r["target_name"], r["data_foundry_uri"],
                           int(r["parquet_bytes"]))
        for r in csv.DictReader(text.splitlines())
    )


#: BeyondArena's 142 datasets. Source in the module docstring.
BEYONDARENA_DATASETS: tuple[BeyondArenaDataset, ...] = _read_beyondarena()

#: BeyondArena's 142 dataset names, sorted.
BEYONDARENA: tuple[str, ...] = tuple(d.name for d in BEYONDARENA_DATASETS)

#: Shortest normalised name for which containment, not only equality, counts as a match:
#: BeyondArena's ``blood_transfusion`` is TabArena's ``blood-transfusion-service-center``, while
#: very short names (``mic``, ``jm1``) would match unrelated datasets by accident.
MIN_CONTAINED = 6


def normalise_name(name: str) -> str:
    """Lower-case a dataset name and keep only letters and digits, so separators never matter."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def same_dataset_name(a: str, b: str) -> bool:
    """Whether two dataset names plausibly denote one dataset.

    Args:
        a: A dataset name in any separator convention.
        b: Another dataset name.

    Returns:
        True when the normalised names are equal, or when the shorter one (at least
        :data:`MIN_CONTAINED` characters) is contained in the longer.
    """
    x, y = normalise_name(a), normalise_name(b)
    short, long_ = sorted((x, y), key=len)
    return x == y or (len(short) >= MIN_CONTAINED and short in long_)


def name_clashes(name: str) -> list[str]:
    """Every TabArena-v0.1 or BeyondArena dataset whose name matches ``name``.

    Args:
        name: The candidate dataset's name.

    Returns:
        Descriptions such as ``"TabArena-v0.1 credit-g"``, empty when there is no clash.
    """
    hits = [f"TabArena-v0.1 {t}" for _, t in TABARENA_V0_1 if same_dataset_name(name, t)]
    return hits + [f"BeyondArena {b}" for b in BEYONDARENA if same_dataset_name(name, b)]


#: Dataset names no arm-selection tool may use (DR4.1): the two TabArena-v0.1 bankruptcy panels
#: under TabArena's names and under this project's loaders' names
#: (``fintfm.evaluation.datasets``: ``polish-bankruptcy-<h>y``, ``taiwan-bankruptcy``). Matched
#: with :func:`same_dataset_name`, so ``polish_bankruptcy.zip`` and ``polish-bankruptcy-3y``
#: both hit; the loaders' spellings are listed because ``taiwan`` is not contained in
#: ``taiwanese``.
SELECTION_FORBIDDEN: tuple[str, ...] = (
    "polish_companies_bankruptcy",
    "taiwanese_bankruptcy_prediction",
    "polish-bankruptcy",
    "taiwan-bankruptcy",
)

#: The command-line flag that lets a tool read a forbidden dataset, for reporting only.
REPORTING_FLAG = "--allow-tabarena-for-reporting"


class SelectionForbiddenError(ValueError):
    """An arm-selection tool was handed a dataset in :data:`SELECTION_FORBIDDEN`."""


def selection_forbidden(name: str) -> str | None:
    """The :data:`SELECTION_FORBIDDEN` entry ``name`` matches, or None.

    Args:
        name: A dataset name, an archive name or a run's recorded ``dataset`` field.

    Returns:
        The matching entry, or None when the dataset may be used for selection.
    """
    return next((f for f in SELECTION_FORBIDDEN if same_dataset_name(name, f)), None)


def guard_selection(names: Iterable[str], allow_reporting: bool, tool: str) -> list[str]:
    """Refuse forbidden datasets in an arm-selection tool, or flag them as reporting only.

    Args:
        names: The datasets the tool is about to use.
        allow_reporting: Whether :data:`REPORTING_FLAG` was passed.
        tool: The tool's name, for the message.

    Returns:
        The forbidden names found, sorted; empty when there are none. A caller that gets a
        non-empty list back (only possible with ``allow_reporting``) labels its output as
        reporting only.

    Raises:
        SelectionForbiddenError: If a forbidden dataset is present and ``allow_reporting`` is
            false.
    """
    hits = sorted({n for n in names if selection_forbidden(n)})
    if hits and not allow_reporting:
        raise SelectionForbiddenError(
            f"{tool}: {', '.join(hits)}: TabArena-v0.1 bankruptcy panel(s), and choosing an arm "
            "on one biases the TabArena number this project reports (design review DR4.1). "
            f"Pass {REPORTING_FLAG} to report on it; the result must not select anything."
        )
    return hits


#: Where the TabArena fork's runs leave TabArena-v0.1's datasets: openml-python's default
#: cache, ``<id>/dataset_<id>.pq`` beside ``description.xml``. Only :func:`fetch_tabarena`
#: writes to it; everything else reads.
TABARENA_OPENML_CACHE = Path.home() / ".openml" / "org" / "openml" / "www" / "datasets"

#: Significant figures numeric cells are rounded to before hashing, so a float32 copy of a
#: float64 column hashes the same.
SIGNIFICANT = 5

#: Fewest numeric cells a row needs for a row hash; a row of one number collides too easily.
MIN_ROW_CELLS = 2

#: Fewest distinct values for a numeric column to be fingerprinted; a 0/1 column would match
#: any other 0/1 column with the same count.
MIN_DISTINCT_NUMERIC = 10

#: Fewest levels for a categorical column to be fingerprinted by its level counts.
MIN_DISTINCT_CATEGORIES = 3

#: Row share or column share at or above which two datasets are called the same.
OVERLAP_SHARE = 0.5

#: Fewest shared column fingerprints for the column share to count.
MIN_SHARED_COLUMNS = 3


@dataclass(frozen=True)
class ContentFingerprint:
    """What :func:`content_fingerprint` records of one dataset.

    Attributes:
        n_rows: Rows in the dataset.
        rows: Sorted unique 64-bit row hashes, from rows with at least :data:`MIN_ROW_CELLS`
            numeric cells.
        columns: Column fingerprints (hex digests prefixed ``n:`` or ``c:``).
    """

    n_rows: int
    rows: np.ndarray
    columns: frozenset[str]


def _numeric_view(col: pd.Series) -> np.ndarray:
    """A column as floats where its cells are numbers or numeric strings, NaN elsewhere."""
    if pd.api.types.is_bool_dtype(col) or pd.api.types.is_numeric_dtype(col):
        return col.to_numpy(dtype=float, na_value=np.nan)
    return pd.to_numeric(col.astype("object"), errors="coerce").to_numpy(dtype=float)


def _rounded(values: np.ndarray) -> np.ndarray:
    """Finite floats as strings at :data:`SIGNIFICANT` significant figures; others as ``""``."""
    out = np.full(values.shape, "", dtype=object)
    finite = np.isfinite(values)
    out[finite] = np.char.mod(f"%.{SIGNIFICANT}g", values[finite]).astype(object)
    return out


def _digest(text: str) -> str:
    """A short hex digest, stable across processes (unlike ``hash``)."""
    return hashlib.blake2b(text.encode(), digest_size=12).hexdigest()


def content_fingerprint(features: pd.DataFrame) -> ContentFingerprint:
    """Row hashes and column fingerprints of a dataset's features (target excluded).

    A column counts as numeric when every non-missing cell parses as a number, so a category
    column holding ``"1"``, ``"2"`` reads the same as an integer column holding 1, 2. The
    module docstring says what each part is robust to and what it misses.

    Args:
        features: The feature columns only; the caller drops the target, whose encoding the
            re-uploads change.

    Returns:
        The dataset's :class:`ContentFingerprint`.
    """
    numeric: list[np.ndarray] = []
    columns: set[str] = set()
    for name in features.columns:
        num, fingerprint = _column_part(features[name])
        if num is not None:
            numeric.append(num)
        if fingerprint is not None:
            columns.add(fingerprint)
    rows = np.unique(np.asarray(_row_hashes(numeric), dtype=np.uint64))
    return ContentFingerprint(len(features), rows, frozenset(columns))


def _column_part(col: pd.Series) -> tuple[np.ndarray | None, str | None]:
    """One column's contribution to a fingerprint, shared by both fingerprint functions.

    Args:
        col: The whole column, in pandas' reading of it.

    Returns:
        ``(numeric, fingerprint)``. ``numeric`` is the column as floats when every non-missing
        cell parses as a finite number (it then feeds the row hashes), else None.
        ``fingerprint`` is its ``n:`` or ``c:`` column fingerprint, or None when it has too few
        distinct values or levels.
    """
    num = _numeric_view(col)
    present = col.notna().to_numpy()
    if present.any() and np.isfinite(num[present]).all():
        vals = np.sort(num[np.isfinite(num)])
        if len(np.unique(vals)) >= MIN_DISTINCT_NUMERIC:
            return num, "n:" + _digest("\x1f".join(_rounded(vals)))
        return num, None
    counts = col.astype("object").value_counts(dropna=True)
    if len(counts) >= MIN_DISTINCT_CATEGORIES:
        return None, "c:" + _digest(",".join(map(str, sorted(counts.tolist()))))
    return None, None


def _row_hashes(numeric: list[np.ndarray]) -> list[int]:
    """The row hash of each row with at least :data:`MIN_ROW_CELLS` finite cells.

    Args:
        numeric: The numeric columns' float views, all of one length (see
            :func:`_column_part`); empty for a table with none.

    Returns:
        One 64-bit hash per qualifying row, in row order, duplicates kept.
    """
    hashes: list[int] = []
    # in row chunks: a 70,000 x 784 table as one array of strings would need gigabytes
    for start in range(0, len(numeric[0]) if numeric else 0, 2048):
        cells = np.column_stack([_rounded(v[start : start + 2048]) for v in numeric])
        for row in cells:
            kept = sorted(c for c in row if c)
            if len(kept) >= MIN_ROW_CELLS:
                hashes.append(int(_digest("\x1f".join(kept))[:16], 16))
    return hashes


#: Cells (rows times numeric columns) per row batch in :func:`parquet_fingerprint`; about
#: 16 MB of float64, so a 989-column reference is read some 2,000 rows at a time.
BATCH_CELLS = 2_000_000


def _feature_names(path: Path, target: str | None) -> list[str]:
    """A parquet file's feature columns: every column except ``target`` and pandas' index.

    Raises:
        ValueError: If ``target`` is given and is not a column of the file.
    """
    import pyarrow.parquet as pq

    schema = pq.read_schema(path)
    index = {c for c in (schema.pandas_metadata or {}).get("index_columns", [])
             if isinstance(c, str)}
    names = [n for n in schema.names if n not in index]
    if target is not None and target not in names:
        raise ValueError(f"{path}: no target column {target!r}")
    return [n for n in names if n != target]


def parquet_fingerprint(
    path: Path, target: str | None, keep_rows: np.ndarray | None = None
) -> ContentFingerprint:
    """:func:`content_fingerprint` of a parquet file's features, read without loading the file.

    Two passes over the file with pyarrow. The first reads one column at a time, converted to
    pandas as :func:`pandas.read_parquet` would, to decide whether it is numeric and to build its
    column fingerprint (:func:`_column_part`, the code :func:`content_fingerprint` runs). The
    second reads only the numeric columns in row batches of about :data:`BATCH_CELLS` cells and
    hashes each row (:func:`_row_hashes`). Peak memory is one column plus one batch, not the
    file: the 3.5 GB BeyondArena ``maps_router_eta`` reads as 1.25 million rows of one column,
    then about 2,000 rows of all 989.

    With ``keep_rows`` None the result equals ``content_fingerprint`` of
    ``pandas.read_parquet(path)`` with the target dropped (``tests/test_overlap_report.py``
    holds that equality). A row share needs only the candidate hashes found in a reference, so
    passing the union of every candidate's row hashes as ``keep_rows`` keeps the result small
    without changing any :func:`content_match` against those candidates.

    Args:
        path: The parquet file.
        target: The target column, left out like :func:`content_fingerprint`'s caller drops
            it; None keeps every column.
        keep_rows: Row hashes worth keeping (any order, uint64); None keeps them all.

    Returns:
        The file's :class:`ContentFingerprint`; ``n_rows`` always counts every row, and with
        ``keep_rows`` given ``rows`` holds only the hashes in it.

    Raises:
        ValueError: If ``target`` is given and is not a column of the file.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    names = _feature_names(path, target)
    pf = pq.ParquetFile(path, buffer_size=1 << 20, pre_buffer=False)
    numeric: list[str] = []
    columns: set[str] = set()
    for name in names:
        col = pf.read(columns=[name]).to_pandas().iloc[:, 0]
        num, fingerprint = _column_part(col)
        del col
        if num is not None:
            numeric.append(name)
        if fingerprint is not None:
            columns.add(fingerprint)
    keep = None if keep_rows is None else np.unique(np.asarray(keep_rows, dtype=np.uint64))
    found: list[np.ndarray] = []
    if numeric:
        batch_rows = max(64, BATCH_CELLS // len(numeric))
        for batch in pf.iter_batches(batch_size=batch_rows, columns=numeric):
            frame = pa.Table.from_batches([batch]).to_pandas()
            views = [_numeric_view(frame.iloc[:, i]) for i in range(frame.shape[1])]
            hashes = np.asarray(_row_hashes(views), dtype=np.uint64)
            if keep is not None:
                hashes = hashes[np.isin(hashes, keep)]
            found.append(np.unique(hashes))
    rows = np.unique(np.concatenate(found)) if found else np.asarray([], dtype=np.uint64)
    return ContentFingerprint(pf.metadata.num_rows, rows, frozenset(columns))


@dataclass(frozen=True)
class ContentMatch:
    """How much of one candidate dataset a reference dataset holds.

    Attributes:
        reference: The reference's label, e.g. ``"TabArena-v0.1 46918 credit-g"``.
        row_share: Share of the candidate's row hashes found in the reference; NaN when the
            candidate has none.
        column_share: Share of the candidate's column fingerprints found in the reference;
            NaN when it has none.
        shared_columns: Column fingerprints in common.
    """

    reference: str
    row_share: float
    column_share: float
    shared_columns: int

    @property
    def overlaps(self) -> bool:
        """Whether the two are the same dataset by the thresholds in the module docstring."""
        rows = self.row_share >= OVERLAP_SHARE
        cols = self.shared_columns >= MIN_SHARED_COLUMNS and self.column_share >= OVERLAP_SHARE
        return bool(rows or cols)


def content_match(
    candidate: ContentFingerprint, reference: ContentFingerprint, label: str
) -> ContentMatch:
    """Compare a candidate's fingerprint against one reference's.

    Args:
        candidate: The dataset being admitted.
        reference: A TabArena dataset's fingerprint.
        label: The reference's label for the report.

    Returns:
        The :class:`ContentMatch`.
    """
    row_share = (
        float(np.isin(candidate.rows, reference.rows).mean()) if len(candidate.rows) else np.nan
    )
    shared = len(candidate.columns & reference.columns)
    col_share = shared / len(candidate.columns) if candidate.columns else np.nan
    return ContentMatch(label, row_share, col_share, shared)


def _target_attribute(description: Path) -> str | None:
    """The ``default_target_attribute`` of an openml-python ``description.xml``, if any."""
    try:
        root = ET.parse(description).getroot()
    except (OSError, ET.ParseError):
        return None
    for el in root.iter():
        if el.tag.endswith("default_target_attribute") and el.text:
            return el.text.strip()
    return None


class ReferenceUnavailable(RuntimeError):
    """A reference dataset is not available locally, or could not be fetched; the message
    says why, for the report's per-reference status."""


def tabarena_label(dataset_id: int, name: str) -> str:
    """A TabArena-v0.1 dataset's label in reports, e.g. ``"TabArena-v0.1 46918 credit-g"``."""
    return f"TabArena-v0.1 {dataset_id} {name}"


def fetch_tabarena(dataset_id: int, cache_dir: Path = TABARENA_OPENML_CACHE) -> Path:
    """Download one TabArena-v0.1 dataset into openml-python's cache (network).

    openml-python (the ``overlap`` extra) writes ``<id>/dataset_<id>.pq`` and
    ``description.xml`` in the layout the fork's runs left, so a fetched dataset reads exactly
    like one the fork cached. Data only; no task, split or quality files.

    Args:
        dataset_id: A TabArena-v0.1 OpenML dataset id (:data:`TABARENA_V0_1`).
        cache_dir: openml-python's dataset cache, which must end in
            ``org/openml/www/datasets`` (the layout for the default OpenML server).

    Returns:
        The dataset's parquet file.

    Raises:
        ReferenceUnavailable: If openml-python is not installed, the download fails, or the
            server offered no parquet copy (openml-python then falls back to ARFF, which is not
            read here).
    """
    if tuple(cache_dir.parts[-4:]) != ("org", "openml", "www", "datasets"):
        raise ReferenceUnavailable(f"{cache_dir} is not an openml-python dataset cache "
                                   "(<root>/org/openml/www/datasets)")
    try:
        import openml
    except ImportError as exc:
        raise ReferenceUnavailable("openml-python not installed (uv sync --extra overlap, with "
                                   "the session's other extras)") from exc
    openml.config.set_root_cache_directory(cache_dir.parents[3])
    try:
        openml.datasets.get_dataset(dataset_id, download_data=True, download_qualities=False,
                                    download_features_meta_data=False)
    except Exception as exc:  # any failure is one reference's status
        raise ReferenceUnavailable(f"openml fetch failed ({type(exc).__name__}: {exc})") from exc
    path = cache_dir / str(dataset_id) / f"dataset_{dataset_id}.pq"
    if not path.exists():
        raise ReferenceUnavailable("openml-python fetched no parquet copy (ARFF fallback)")
    return path


def tabarena_parquet(
    dataset_id: int, cache_dir: Path = TABARENA_OPENML_CACHE, fetch: bool = False
) -> tuple[Path, str]:
    """One TabArena-v0.1 dataset's parquet file and target column.

    Args:
        dataset_id: The OpenML dataset id.
        cache_dir: openml-python's dataset cache.
        fetch: Download the dataset with :func:`fetch_tabarena` when it is not cached.

    Returns:
        ``(parquet, target)``, the target read from the cached ``description.xml``.

    Raises:
        ReferenceUnavailable: If the dataset is not cached (and ``fetch`` is false or fails)
            or its description names no target.
    """
    path = cache_dir / str(dataset_id) / f"dataset_{dataset_id}.pq"
    if not path.exists():
        if not fetch:
            raise ReferenceUnavailable(f"not in the OpenML cache {cache_dir}; pass --fetch")
        path = fetch_tabarena(dataset_id, cache_dir)
    target = _target_attribute(path.parent / "description.xml")
    if target is None:
        raise ReferenceUnavailable(f"{path.parent / 'description.xml'} names no target")
    return path, target


def beyondarena_parquet(dataset: BeyondArenaDataset, fetch: bool = False) -> Path:
    """One BeyondArena dataset's parquet file, from the Hugging Face cache.

    ``huggingface_hub.hf_hub_download`` at :data:`BEYONDARENA_REVISION`, one file at a time into
    the normal Hugging Face cache, so only the dataset's own ``dataset.parquet`` and its task
    metadata are fetched (never the text-embedding parquet files beside some of them). The task
    metadata's ``target_column_name`` must equal the fork's ``target_name``.

    Args:
        dataset: The dataset, from :data:`BEYONDARENA_DATASETS`.
        fetch: Download what the cache lacks (network); otherwise read the cache only.

    Returns:
        The local path of the dataset's parquet file.

    Raises:
        ReferenceUnavailable: If huggingface_hub is not installed, a file is not cached (and
            ``fetch`` is false or the download fails), or the targets disagree.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ReferenceUnavailable("huggingface_hub not installed (uv sync --extra overlap, "
                                   "with the session's other extras)") from exc

    def get(filename: str) -> Path:
        """One file of the repository at the pinned revision, from the cache or the Hub."""
        try:
            return Path(hf_hub_download(BEYONDARENA_REPO, filename, repo_type="dataset",
                                        revision=BEYONDARENA_REVISION,
                                        local_files_only=not fetch))
        except Exception as exc:  # any failure is one reference's status
            where = "fetch failed" if fetch else "not in the Hugging Face cache; pass --fetch"
            raise ReferenceUnavailable(f"{filename}: {where} ({type(exc).__name__})") from exc

    meta = json.loads(get(dataset.task_metadata_file).read_text(encoding="utf-8"))
    if meta.get("target_column_name") != dataset.target:
        raise ReferenceUnavailable(f"task metadata names target {meta.get('target_column_name')!r},"
                                   f" the fork's metadata {dataset.target!r}")
    return get(dataset.parquet_file)


def load_reference_fingerprints(
    cache_dir: Path = TABARENA_OPENML_CACHE, keep_rows: np.ndarray | None = None
) -> tuple[dict[str, ContentFingerprint], list[str]]:
    """Fingerprint every TabArena-v0.1 dataset the local OpenML cache holds (read only).

    The ``--row-hash`` checks' references. The full comparison, BeyondArena included, is the
    report in :mod:`fintfm.evaluation.overlap_report`.

    Args:
        cache_dir: openml-python's dataset cache.
        keep_rows: Passed to :func:`parquet_fingerprint`; None keeps every row hash.

    Returns:
        ``(fingerprints, notes)``: fingerprints keyed by a label such as
        ``"TabArena-v0.1 46918 credit-g"``, and notes on what could not be compared (datasets
        not cached, BeyondArena, no parquet reader). An empty dict means the check is skipped,
        and the notes say why.
    """
    beyond = ("BeyondArena: not compared by contents here; the full report "
              "(python -m fintfm.evaluation.overlap --report) compares it, and its names are "
              "checked by name_clashes")
    if not cache_dir.is_dir():
        return {}, [f"TabArena-v0.1: no local cache at {cache_dir}; row-hash check skipped", beyond]
    try:
        import pyarrow  # noqa: F401  pandas' parquet engine, from the bench extra
    except ImportError:
        return {}, [("TabArena-v0.1: pyarrow not installed (uv sync --extra bench); row-hash "
                     "check skipped"), beyond]
    out: dict[str, ContentFingerprint] = {}
    missing: list[str] = []
    for dataset_id, name in TABARENA_V0_1:
        pq = cache_dir / str(dataset_id) / f"dataset_{dataset_id}.pq"
        if not pq.exists():
            missing.append(f"{dataset_id} {name}")
            continue
        target = _target_attribute(cache_dir / str(dataset_id) / "description.xml")
        if target is not None and target not in _feature_names(pq, None):
            target = None  # as before: a target the file lacks drops nothing
        out[tabarena_label(dataset_id, name)] = parquet_fingerprint(pq, target, keep_rows)
    notes = [f"TabArena-v0.1: {len(out)} of {len(TABARENA_V0_1)} datasets read from {cache_dir}"]
    if missing:
        notes.append(f"TabArena-v0.1: not cached, so not compared by contents: "
                     f"{', '.join(missing)}")
    return out, [*notes, beyond]


def best_content_match(
    candidate: ContentFingerprint, references: dict[str, ContentFingerprint]
) -> ContentMatch | None:
    """The reference holding most of the candidate (an overlapping one first), or None."""
    matches = [content_match(candidate, ref, label) for label, ref in references.items()]
    if not matches:
        return None
    return max(matches, key=lambda m: (m.overlaps, np.nan_to_num(m.row_share),
                                       np.nan_to_num(m.column_share)))


def row_hash_report(
    datasets: Iterable[tuple[str, Callable[[], pd.DataFrame | None]]],
    references: dict[str, ContentFingerprint],
    print_fn: Callable[[str], None] = print,
) -> list[str]:
    """Fingerprint each dataset, print its best reference match, return the overlapping names.

    Args:
        datasets: ``(name, load)`` pairs; ``load`` returns the features with the target
            dropped, or None when the dataset could not be fetched (printed, not counted).
        references: From :func:`load_reference_fingerprints`.
        print_fn: Where report lines go.

    Returns:
        The names that overlap a reference.
    """
    hits: list[str] = []
    for name, load in datasets:
        features = load()
        if features is None:
            print_fn(f"{name:<34} NOT CHECKED (could not be fetched)")
            continue
        fp = content_fingerprint(features)
        match = best_content_match(fp, references)
        head = f"{name:<34} {fp.n_rows:>7} rows {len(fp.rows):>7} hashed {len(fp.columns):>4} cols"
        if match is None:
            print_fn(f"{head}  no reference")
            continue
        flag = "OVERLAP" if match.overlaps else "clean"
        print_fn(f"{head}  {flag:<7} best {match.reference}: row share {match.row_share:.3f}, "
                 f"column share {match.column_share:.3f} ({match.shared_columns} shared)")
        if match.overlaps:
            hits.append(name)
    return hits


#: The positive control every content check runs: OpenML's original credit-g (dataset 31),
#: whose re-upload is TabArena-v0.1's 46918 (:data:`CONTROL_REFERENCE`). A check that does not
#: flag it is not working, and nothing else it reports counts.
CONTROL_DATASET_ID = 31

#: The control's name in reports.
CONTROL_NAME = "credit-g (OpenML 31)"

#: The reference the control must overlap.
CONTROL_REFERENCE = tabarena_label(46918, "credit-g")


if __name__ == "__main__":
    # The report lives in its own module, which imports this one by its package name; running
    # it from here keeps the documented command short.
    from fintfm.evaluation.overlap_report import main

    raise SystemExit(main())
