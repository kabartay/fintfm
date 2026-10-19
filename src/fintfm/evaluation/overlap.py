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
  BeyondArena_tasks_metadata.csv``, column ``dataset_name`` (142 datasets; no OpenML ids).

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
with at least :data:`MIN_SHARED_COLUMNS` shared). The references are whatever TabArena-v0.1
datasets the local OpenML cache holds (:data:`TABARENA_OPENML_CACHE`, written by the fork's
runs; read only), so the check covers only those and names the ones it lacked. BeyondArena
lists no OpenML ids, so no local copy can be matched to it and it is not compared by contents.
A dataset subsampled or with its numeric columns transformed on re-upload passes both signals.

**Selection (DR4.1).** The Polish and Taiwanese bankruptcy panels are TabArena-v0.1 datasets
(``polish_companies_bankruptcy``, ``taiwanese_bankruptcy_prediction``) as well as this
project's own credit panels. They may be reported on, never used to choose an arm:
:data:`SELECTION_FORBIDDEN` and :func:`guard_selection` are the one rule every arm-selection
tool calls, overridden only by :data:`REPORTING_FLAG`.
"""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable
from dataclasses import dataclass
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

#: BeyondArena's 142 dataset names, whitespace-separated. Source in the module docstring.
_BEYONDARENA_TEXT = """
5g_energy_consumption acquire_valued_shoppers_challenge airfoil_self_noise
allstate_claims_severity amazon_employee_access amex_non_iid_1m anes_voting_2026 aps_failure
asp_potassco_classification audiology_diagnosis bad_customer_detection bank_customer_churn
bank_marketing biogeographical_ancestry_prediction biomechanical_orthopaedic_prediction
bioresponse blood_tests_drink_prediction blood_transfusion body_density_prediction
california_house_prices_2020 cardiotocography churn cirrhosis_patient_survival_prediction
climate_model_weather_forecasting_1m clock_protein_toxicity coffee_rating_prediction coil_2000
concrete_compressive_strength consumer_complaints_1m cooking_time_1m covertype credit_approval
credit_card_clients_default credit_g customer_satisfaction_in_airline delivery_eta_1m
dementia_prediction diabetes_130_us diamonds drug_induced_autoimmunity_prediction
early_learning_predictors early_stage_diabetes_risk_prediction ecoli_proteins
ecommerce_shipping electric_motor_temperature_prediction emscad eryhemato_squamous_disease
fiat_500 fitness_club food_delivery_time forensic_glass_identification forest_fires
gallstone_disease garments_worker_productivity ghanas_indigenous_intel give_me_some_credit
hazelnut_spread_contaminant_detection healthcare_insurance_expenses heart_disease_cleveland
heart_disease_hungary heart_disease_va_long_beach heart_failure_followup_survival heloc
hepatitis_c_prediction hepatitis_survival_prediction hiva_agnostic home_credit_default_risk
home_credit_default_stability_1m homeq_default_prediction homesite_quote_conversion
horse_colic_survival hotel_booking_demand houses hr_analytics ieee_fraud_detection
immoscout_german_house_prices in_vehicle_coupon_recommendation indian_liver_patient_dataset
iranian_churn jm1 kdd_cup_09_appetency kick kickstarter labour_inspection_compliance
lending_club_1m ljubljana_breast_cancer ljubljana_primary_tumor lung_cancer
lung_cancer_epithelial_genexp maps_router_eta_1m marketing_campaign maternal_health_risk
mercari_price_suggestion_1m mercedes_benz_greener_manufacturing miami_housing mic
mice_protein_trisomy_discriminant micro_mass musk mutual_funds_india
naticusdroid_android_permissions_dataset obesity_estimation
online_shoppers_purchasing_intention_dataset otto_group_product_classification_challenge
pancreatic_cancer_mouse_detection parkinsons_biomedical_voice_measurements
physiochemical_protein polish_companies_bankruptcy porto_seguro
predict_students_dropout_and_academic_success prostate_cancer_detection
pva_revenue_prediction_kddcup98 qsar_aquatic_toxicity qsar_biodeg qsar_fish_toxicity
qsar_tid_11 regensburg_pediatric_appendicitis rossmann_store_sales
santander_customer_satisfaction santander_customer_transaction_prediction
santander_transaction_value sat11_hand_algo_runtime sberbank_housing_market_forecasting sdss_17
seismic_bumps sepsis_prediction_1m sepsis_survival_minimal_clinical_records sf_permit_time
south_africa_coronary_heart_disease splice student_portuguese_performance superconductivity
taiwanese_bankruptcy_prediction telemonitoring_parkinsons_biomedical_voice_measurements
thyroid_discordant tour_travels_churn video_game_fps_prediction
video_transcoding_time_prediction website_phishing wids_diabetes_mellitus wine_quality
wine_world_cost
"""
BEYONDARENA: tuple[str, ...] = tuple(_BEYONDARENA_TEXT.split())

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
#: cache, ``<id>/dataset_<id>.pq`` beside ``description.xml``. Read only.
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
        col = features[name]
        num = _numeric_view(col)
        present = col.notna().to_numpy()
        if present.any() and np.isfinite(num[present]).all():
            numeric.append(num)
            vals = np.sort(num[np.isfinite(num)])
            if len(np.unique(vals)) >= MIN_DISTINCT_NUMERIC:
                columns.add("n:" + _digest("\x1f".join(_rounded(vals))))
        else:
            counts = col.astype("object").value_counts(dropna=True)
            if len(counts) >= MIN_DISTINCT_CATEGORIES:
                columns.add("c:" + _digest(",".join(map(str, sorted(counts.tolist())))))
    hashes: list[int] = []
    # in row chunks: a 70,000 x 784 table as one array of strings would need gigabytes
    for start in range(0, len(features) if numeric else 0, 2048):
        cells = np.column_stack([_rounded(v[start : start + 2048]) for v in numeric])
        for row in cells:
            kept = sorted(c for c in row if c)
            if len(kept) >= MIN_ROW_CELLS:
                hashes.append(int(_digest("\x1f".join(kept))[:16], 16))
    rows = np.unique(np.asarray(hashes, dtype=np.uint64))
    return ContentFingerprint(len(features), rows, frozenset(columns))


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


def load_reference_fingerprints(
    cache_dir: Path = TABARENA_OPENML_CACHE,
) -> tuple[dict[str, ContentFingerprint], list[str]]:
    """Fingerprint every TabArena-v0.1 dataset the local OpenML cache holds (read only).

    Args:
        cache_dir: openml-python's dataset cache.

    Returns:
        ``(fingerprints, notes)``: fingerprints keyed by a label such as
        ``"TabArena-v0.1 46918 credit-g"``, and notes on what could not be compared (datasets
        not cached, BeyondArena, no parquet reader). An empty dict means the check is skipped,
        and the notes say why.
    """
    beyond = ("BeyondArena: not compared by contents (it lists no OpenML ids, so no local copy "
              "can be matched to it); its names are checked by name_clashes")
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
        frame = pd.read_parquet(pq)
        target = _target_attribute(cache_dir / str(dataset_id) / "description.xml")
        if target in frame.columns:
            frame = frame.drop(columns=[target])
        out[f"TabArena-v0.1 {dataset_id} {name}"] = content_fingerprint(frame)
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
