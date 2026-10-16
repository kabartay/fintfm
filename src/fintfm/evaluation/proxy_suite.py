"""A proxy Elo suite outside TabArena and BeyondArena (design review 2026-10-16, task DR4.2).

TabArena is the benchmark this project reports on, so choosing arms on it biases the number it
reports. The proxy is a second set of OpenML tasks, disjoint from TabArena-v0.1 and BeyondArena,
on which arms can be ranked before the TabArena run that reports them. It is a decision tool
only after its Elo has been shown to rank checkpoints the way TabArena-Lite does (DR4.2's
Verify: Spearman 0.8 or higher across six or more existing checkpoints). Until then, nothing is
decided on it.

**What this module is.** The candidate pool (every task of OpenML-CC18, study 99, and
OpenML-CTR23, study 353), the filters that remove candidates, and the selection that turns what
is left into the suite. ``--dry-run`` prints all of it. It does not fetch data, fit anything or
score anything.

**The filters, in the order the dry run reports them:**

1. *TabArena-v0.1 dataset id* (`fintfm.evaluation.overlap.TABARENA_V0_1`). Catches nothing
   today: TabArena's ids are its own re-uploads (46904 onwards), not these suites' originals.
2. *Name* against TabArena-v0.1 and BeyondArena (`fintfm.evaluation.overlap.name_clashes`).
   Containment over-matches on purpose: ``brazilian_houses`` (Brazilian rentals) is excluded
   for containing TabArena's ``houses`` (California housing), and ``vehicle`` for BeyondArena's
   ``in_vehicle_coupon_recommendation``. A missed overlap costs more than a lost task.
3. *Same source under another name* (:data:`SAME_SOURCE`), found by comparing row and feature
   counts and source URLs against the fork's TabArena and BeyondArena metadata. A row-hash
   check would be stronger and needs the data; it has not been run.
4. *Licence at the original source* (:data:`SOURCE_LICENCES`). Each candidate that passes
   filters 1-3 was traced from its OpenML description to the page it came from (a UCI
   repository page, a Kaggle dataset, a Zenodo record, a GitHub repository, a paper or a
   university site), and that page's licence was read on 2026-10-16 and recorded with its URL,
   the licence as stated, the verdict and the reason. A task is admitted only by an entry
   marked include, and only CC BY (any version), CC0, public domain or MIT pass
   (:func:`licence_verdict`). Anything with no licence on the page, an unreachable page, a
   non-commercial or copyleft term, or a permissive licence over data scraped from a site with
   its own terms is excluded. A candidate with no entry is excluded as unverified.

**Why OpenML's own field does not decide (owner decision, 2026-10-16).** Every OpenML-CC18
dataset carries ``Public``, the upload default rather than a named licence, and OpenML's terms
of use (``https://github.com/openml/docs/blob/main/docs/intro/terms.md``) grant each user a
licence "for their own research purposes, in accordance with any licences granted by you". That
is not a commercial grant. The field is still recorded per candidate
(``https://www.openml.org/api/v1/json/data/<dataset_id>``, fetched 2026-10-16), and
``--verify-licences`` re-fetches it and reports any change, but the verdict comes from the
source. The source check is not re-run automatically; a re-check means reading each
:data:`SOURCE_LICENCES` URL again and updating its date.

**What the check left (2026-10-16).** Of 77 candidates that pass the overlap filters, 53 have a
verified source licence (49 from UCI pages, all of which state CC BY 4.0; pc1 from its Zenodo
record, CC BY 4.0; numerai28.6 and kings_county from Kaggle, CC0; Fashion-MNIST from GitHub,
MIT) and 24 are excluded. The suite is 18 binary, 8 multiclass and 9 regression tasks (35), of
which 16, 7 and 9 (32) are within the 136-feature cap. Five of the eight multiclass tasks are
feature sets of one UCI collection (Multiple Features); lowest-task-id selection picks them and
nothing here corrects for it.

**Intended runner (not yet written).** The suite goes through TabArena's custom-task pipeline,
the pattern of the fork's ``examples/advanced/run_any_openml_task.py``: the task ids from
``--task-ids`` into ``task_metadata_collection_from_openml(tids=...)``, three splits via
``subset_tasks(split_indices=[0, 1, 2])``, a ``TabArenaV0pt1ExperimentBundle`` of FinTFM beside
the RF, ExtraTrees, Linear, KNN and GBDT defaults, and
``AbstractArenaContext(methods=[]).compare(calibration_method="RF (default)")`` for Elo
anchored at RF (default) = 1000. Run from the TabArena fork's environment, for example::

    uv run --frozen python -m fintfm.evaluation.proxy_suite --task-ids   # in this repo
    FINTFM_PROXY_TIDS=<ids> python -P run_fintfm_proxy.py                # in the fork, later

**Evaluation only.** No OpenML data trains anything here; ``CLAUDE.md``'s licensing boundary
bans third-party training data and permits evaluation.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TextIO
from urllib.error import URLError

from fintfm.evaluation.overlap import TABARENA_V0_1, name_clashes

#: Problem types in the order the suite lists them.
KINDS: tuple[str, ...] = ("binary", "multiclass", "regression")

#: DR4.2's target mix, which mirrors TabArena-v0.1's own 30 / 8 / 13. A smaller count is
#: expected when the filters leave fewer candidates of a kind; it is reported, never padded.
TARGET_COUNTS: dict[str, int] = {"binary": 30, "multiclass": 8, "regression": 13}

#: The current checkpoints' feature cap, reported beside each task rather than used as a
#: filter: the proxy should rank checkpoints on the same kind of coverage TabArena measures.
FEATURE_CAP = 136

#: Where each candidate's OpenML licence field was read, formatted with the dataset id.
LICENCE_SOURCE = "https://www.openml.org/api/v1/json/data/{dataset_id} (field licence)"

#: Date the candidate metadata and licences were fetched from the OpenML API.
FETCHED = "2026-10-16"

#: Date every :data:`SOURCE_LICENCES` entry was read at its source.
SOURCE_CHECKED = "2026-10-16"


@dataclass(frozen=True)
class Candidate:
    """One OpenML task considered for the proxy suite.

    Attributes:
        suite: ``"CC18"`` or ``"CTR23"``, the OpenML benchmark suite it came from.
        task_id: OpenML task id, which fixes the target and the split procedure.
        dataset_id: OpenML dataset id the task is defined on.
        name: OpenML dataset name.
        kind: ``"binary"``, ``"multiclass"`` or ``"regression"``.
        n_rows: OpenML ``NumberOfInstances``.
        n_features: OpenML ``NumberOfFeatures`` minus the target.
        n_classes: OpenML ``NumberOfClasses``; 0 for regression.
        licence: the dataset's OpenML ``licence`` field, verbatim. Recorded for drift checks
            only; :data:`SOURCE_LICENCES` decides admission.
    """

    suite: str
    task_id: int
    dataset_id: int
    name: str
    kind: str
    n_rows: int
    n_features: int
    n_classes: int
    licence: str


def _c(
    suite: str, task_id: int, dataset_id: int, name: str, kind: str, n_rows: int,
    n_features: int, n_classes: int, licence: str,
) -> Candidate:
    """Positional constructor that keeps the :data:`CANDIDATES` table one row per task."""
    return Candidate(suite, task_id, dataset_id, name, kind, n_rows, n_features, n_classes, licence)


#: Every task of OpenML-CC18 (72) and OpenML-CTR23 (35), with the metadata and licence the
#: OpenML API returned on :data:`FETCHED` (``/study/99``, ``/study/353``, ``/task/<id>``,
#: ``/data/<id>``, ``/data/qualities/<id>``).
CANDIDATES: tuple[Candidate, ...] = (
    _c('CC18', 3, 3, 'kr-vs-kp', 'binary', 3196, 36, 2, 'Public'),
    _c('CC18', 6, 6, 'letter', 'multiclass', 20000, 16, 26, 'Public'),
    _c('CC18', 11, 11, 'balance-scale', 'multiclass', 625, 4, 3, 'Public'),
    _c('CC18', 12, 12, 'mfeat-factors', 'multiclass', 2000, 216, 10, 'Public'),
    _c('CC18', 14, 14, 'mfeat-fourier', 'multiclass', 2000, 76, 10, 'Public'),
    _c('CC18', 15, 15, 'breast-w', 'binary', 699, 9, 2, 'Public'),
    _c('CC18', 16, 16, 'mfeat-karhunen', 'multiclass', 2000, 64, 10, 'Public'),
    _c('CC18', 18, 18, 'mfeat-morphological', 'multiclass', 2000, 6, 10, 'Public'),
    _c('CC18', 22, 22, 'mfeat-zernike', 'multiclass', 2000, 47, 10, 'Public'),
    _c('CC18', 23, 23, 'cmc', 'multiclass', 1473, 9, 3, 'Public'),
    _c('CC18', 28, 28, 'optdigits', 'multiclass', 5620, 64, 10, 'Public'),
    _c('CC18', 29, 29, 'credit-approval', 'binary', 690, 15, 2, 'Public'),
    _c('CC18', 31, 31, 'credit-g', 'binary', 1000, 20, 2, 'Public'),
    _c('CC18', 32, 32, 'pendigits', 'multiclass', 10992, 16, 10, 'Public'),
    _c('CC18', 37, 37, 'diabetes', 'binary', 768, 8, 2, 'Public'),
    _c('CC18', 43, 44, 'spambase', 'binary', 4601, 57, 2, 'Public'),
    _c('CC18', 45, 46, 'splice', 'multiclass', 3190, 60, 3, 'Public'),
    _c('CC18', 49, 50, 'tic-tac-toe', 'binary', 958, 9, 2, 'Public'),
    _c('CC18', 53, 54, 'vehicle', 'multiclass', 846, 18, 4, 'Public'),
    _c('CC18', 219, 151, 'electricity', 'binary', 45312, 8, 2, 'Public'),
    _c('CC18', 2074, 182, 'satimage', 'multiclass', 6430, 36, 6, 'Public'),
    _c('CC18', 2079, 188, 'eucalyptus', 'multiclass', 736, 19, 5, 'Public'),
    _c('CC18', 3021, 38, 'sick', 'binary', 3772, 29, 2, 'Public'),
    _c('CC18', 3022, 307, 'vowel', 'multiclass', 990, 12, 11, 'Public'),
    _c('CC18', 3481, 300, 'isolet', 'multiclass', 7797, 617, 26, 'Public'),
    _c('CC18', 3549, 458, 'analcatdata_authorship', 'multiclass', 841, 70, 4, 'Public'),
    _c('CC18', 3560, 469, 'analcatdata_dmft', 'multiclass', 797, 4, 6, 'Public'),
    _c('CC18', 3573, 554, 'mnist_784', 'multiclass', 70000, 784, 10, 'Public'),
    _c('CC18', 3902, 1049, 'pc4', 'binary', 1458, 37, 2, 'Public'),
    _c('CC18', 3903, 1050, 'pc3', 'binary', 1563, 37, 2, 'Public'),
    _c('CC18', 3904, 1053, 'jm1', 'binary', 10885, 21, 2, 'Public'),
    _c('CC18', 3913, 1063, 'kc2', 'binary', 522, 21, 2, 'Public'),
    _c('CC18', 3917, 1067, 'kc1', 'binary', 2109, 21, 2, 'Public'),
    _c('CC18', 3918, 1068, 'pc1', 'binary', 1109, 21, 2, 'Public'),
    _c('CC18', 7592, 1590, 'adult', 'binary', 48842, 14, 2, 'Public'),
    _c('CC18', 9910, 4134, 'Bioresponse', 'binary', 3751, 1776, 2, 'Public'),
    _c('CC18', 9946, 1510, 'wdbc', 'binary', 569, 30, 2, 'Public'),
    _c('CC18', 9952, 1489, 'phoneme', 'binary', 5404, 5, 2, 'Public'),
    _c('CC18', 9957, 1494, 'qsar-biodeg', 'binary', 1055, 41, 2, 'Public'),
    _c('CC18', 9960, 1497, 'wall-robot-navigation', 'multiclass', 5456, 24, 4, 'Public'),
    _c('CC18', 9964, 1501, 'semeion', 'multiclass', 1593, 256, 10, 'Public'),
    _c('CC18', 9971, 1480, 'ilpd', 'binary', 583, 10, 2, 'Public'),
    _c('CC18', 9976, 1485, 'madelon', 'binary', 2600, 500, 2, 'Public'),
    _c('CC18', 9977, 1486, 'nomao', 'binary', 34465, 118, 2, 'Public'),
    _c('CC18', 9978, 1487, 'ozone-level-8hr', 'binary', 2534, 72, 2, 'Public'),
    _c('CC18', 9981, 1468, 'cnae-9', 'multiclass', 1080, 856, 9, 'Public'),
    _c('CC18', 9985, 1475, 'first-order-theorem-proving', 'multiclass', 6118, 51, 6, 'Public'),
    _c('CC18', 10093, 1462, 'banknote-authentication', 'binary', 1372, 4, 2, 'Public'),
    _c('CC18', 10101, 1464, 'blood-transfusion-service-center', 'binary', 748, 4, 2, 'Public'),
    _c('CC18', 14952, 4534, 'PhishingWebsites', 'binary', 11055, 30, 2, 'Public'),
    _c('CC18', 14954, 6332, 'cylinder-bands', 'binary', 540, 39, 2, 'Public'),
    _c('CC18', 14965, 1461, 'bank-marketing', 'binary', 45211, 16, 2, 'Public'),
    _c('CC18', 14969, 4538, 'GesturePhaseSegmentationProcessed', 'multiclass', 9873, 32, 5,
        'Public'),
    _c('CC18', 14970, 1478, 'har', 'multiclass', 10299, 561, 6, 'Public'),
    _c('CC18', 125920, 23381, 'dresses-sales', 'binary', 500, 12, 2, 'Public'),
    _c('CC18', 125922, 40499, 'texture', 'multiclass', 5500, 40, 11, 'Public'),
    _c('CC18', 146195, 40668, 'connect-4', 'multiclass', 67557, 42, 3, 'public'),
    _c('CC18', 146800, 40966, 'MiceProtein', 'multiclass', 1080, 81, 8, 'Public'),
    _c('CC18', 146817, 40982, 'steel-plates-fault', 'multiclass', 1941, 27, 7, 'Public'),
    _c('CC18', 146819, 40994, 'climate-model-simulation-crashes', 'binary', 540, 20, 2, 'Public'),
    _c('CC18', 146820, 40983, 'wilt', 'binary', 4839, 5, 2, 'Public'),
    _c('CC18', 146821, 40975, 'car', 'multiclass', 1728, 6, 4, 'Public'),
    _c('CC18', 146822, 40984, 'segment', 'multiclass', 2310, 19, 7, 'Public'),
    _c('CC18', 146824, 40979, 'mfeat-pixel', 'multiclass', 2000, 240, 10, 'Public'),
    _c('CC18', 146825, 40996, 'Fashion-MNIST', 'multiclass', 70000, 784, 10, 'Public'),
    _c('CC18', 167119, 41027, 'jungle_chess_2pcs_raw_endgame_complete', 'multiclass', 44819, 6, 3,
        'Public'),
    _c('CC18', 167120, 23517, 'numerai28.6', 'binary', 96320, 21, 2, 'Public'),
    _c('CC18', 167121, 40923, 'Devnagari-Script', 'multiclass', 92000, 1024, 46, 'Public'),
    _c('CC18', 167124, 40927, 'CIFAR_10', 'multiclass', 60000, 3072, 10, 'Public'),
    _c('CC18', 167125, 40978, 'Internet-Advertisements', 'binary', 3279, 1558, 2, 'Public'),
    _c('CC18', 167140, 40670, 'dna', 'multiclass', 3186, 180, 3, 'public'),
    _c('CC18', 167141, 40701, 'churn', 'binary', 5000, 20, 2, 'public'),
    _c('CTR23', 361234, 44956, 'abalone', 'regression', 4177, 8, 0, 'CC BY 4.0'),
    _c('CTR23', 361235, 44957, 'airfoil_self_noise', 'regression', 1503, 5, 0, 'CC BY 4.0'),
    _c('CTR23', 361236, 44958, 'auction_verification', 'regression', 2043, 7, 0, 'CC BY 4.0'),
    _c('CTR23', 361237, 44959, 'concrete_compressive_strength', 'regression', 1030, 8, 0,
        'CC BY 4.0'),
    _c('CTR23', 361241, 44963, 'physiochemical_protein', 'regression', 45730, 9, 0, 'CC BY 4.0'),
    _c('CTR23', 361242, 44964, 'superconductivity', 'regression', 21263, 81, 0, 'CC BY 4.0'),
    _c('CTR23', 361243, 44965, 'geographical_origin_of_music', 'regression', 1059, 116, 0,
        'CC BY 4.0'),
    _c('CTR23', 361244, 44966, 'solar_flare', 'regression', 1066, 10, 0, 'CC BY 4.0'),
    _c('CTR23', 361247, 44969, 'naval_propulsion_plant', 'regression', 11934, 14, 0, 'CC BY 4.0'),
    _c('CTR23', 361249, 44971, 'white_wine', 'regression', 4898, 11, 0, 'CC BY 4.0'),
    _c('CTR23', 361250, 44972, 'red_wine', 'regression', 1599, 11, 0, 'CC BY 4.0'),
    _c('CTR23', 361251, 44973, 'grid_stability', 'regression', 10000, 12, 0, 'CC BY 4.0'),
    _c('CTR23', 361252, 44974, 'video_transcoding', 'regression', 68784, 18, 0, 'CC BY 4.0'),
    _c('CTR23', 361253, 44975, 'wave_energy', 'regression', 72000, 48, 0, 'CC BY 4.0'),
    _c('CTR23', 361254, 44976, 'sarcos', 'regression', 48933, 21, 0, 'Public'),
    _c('CTR23', 361255, 44977, 'california_housing', 'regression', 20640, 8, 0, 'Public'),
    _c('CTR23', 361256, 44978, 'cpu_activity', 'regression', 8192, 21, 0, 'Public'),
    _c('CTR23', 361257, 44979, 'diamonds', 'regression', 53940, 9, 0, 'Public'),
    _c('CTR23', 361258, 44980, 'kin8nm', 'regression', 8192, 8, 0, 'Public'),
    _c('CTR23', 361259, 44981, 'pumadyn32nh', 'regression', 8192, 32, 0, 'Public'),
    _c('CTR23', 361260, 44983, 'miami_housing', 'regression', 13932, 15, 0, 'CC0: Public Domain'),
    _c('CTR23', 361261, 44984, 'cps88wages', 'regression', 28155, 6, 0, 'Public'),
    _c('CTR23', 361264, 44987, 'socmob', 'regression', 1156, 5, 0,
        'Non-commercial scholarly and teaching purposes.'),
    _c('CTR23', 361266, 44989, 'kings_county', 'regression', 21613, 21, 0, 'CC 0: Public Domain'),
    _c('CTR23', 361267, 44990, 'brazilian_houses', 'regression', 10692, 9, 0,
        'CC 0: Public Domain'),
    _c('CTR23', 361268, 44992, 'fps_benchmark', 'regression', 24624, 43, 0, 'CC BY'),
    _c('CTR23', 361269, 44993, 'health_insurance', 'regression', 22272, 11, 0, 'GPL (>= 2)'),
    _c('CTR23', 361272, 45012, 'fifa', 'regression', 19178, 28, 0, 'CC0: Public Domain'),
    _c('CTR23', 361616, 41021, 'Moneyball', 'regression', 1232, 14, 0, 'Public'),
    _c('CTR23', 361617, 44960, 'energy_efficiency', 'regression', 768, 8, 0, 'CC BY 4.0'),
    _c('CTR23', 361618, 44962, 'forest_fires', 'regression', 517, 12, 0, 'CC BY 4.0'),
    _c('CTR23', 361619, 44967, 'student_performance_por', 'regression', 649, 30, 0, 'CC BY 4.0'),
    _c('CTR23', 361621, 44970, 'QSAR_fish_toxicity', 'regression', 908, 6, 0, 'CC BY 4.0'),
    _c('CTR23', 361622, 44994, 'cars', 'regression', 804, 17, 0, 'Public'),
    _c('CTR23', 361623, 45402, 'space_ga', 'regression', 3107, 6, 0, 'Public'),
)

#: Candidates that pass the id and name checks but hold a TabArena-v0.1 or BeyondArena dataset
#: (or part of one) under another name, keyed by OpenML dataset id. Found on 2026-10-16 by
#: comparing row and feature counts and source URLs with the fork's
#: ``curated_tabarena_dataset_metadata.csv`` and ``BeyondArena_tasks_metadata.csv``. Entries
#: marked "probable" are excluded to be safe and would need a row hash to confirm.
SAME_SOURCE: dict[int, str] = {
    38: "probable: BeyondArena thyroid_discordant, the same Garavan thyroid collection",
    1480: "BeyondArena indian_liver_patient_dataset (UCI ILPD, 583 rows in both)",
    40670: "TabArena-v0.1 / BeyondArena splice (the StatLog encoding of the same splice data)",
    44971: "TabArena-v0.1 / BeyondArena wine_quality (its white half: 4,898 of 6,497 rows)",
    44972: "TabArena-v0.1 / BeyondArena wine_quality (its red half: 1,599 of 6,497 rows)",
    44967: "BeyondArena student_portuguese_performance (UCI Student Performance, 649 rows)",
    44977: "TabArena-v0.1 houses (California housing, 20,640 rows x 8 features in both)",
    44992: "probable: BeyondArena video_game_fps_prediction, an OpenML 2020 FPS benchmark",
}


@dataclass(frozen=True)
class SourceLicence:
    """A dataset's licence as read at its original source, with the verdict on it.

    Attributes:
        url: The page the licence was read from (or, where noted in ``reason``, the page that
            should have stated one).
        stated: The licence as the page states it, verbatim; empty when the page states none
            or could not be read.
        licence: Short name of the licence, as :func:`licence_verdict` reads it; empty when
            there is none.
        include: True when the dataset may enter the suite.
        reason: Why, in a sentence: what the page says and what it lacks.
        checked: Date the page was read.
    """

    url: str
    stated: str
    licence: str
    include: bool
    reason: str
    checked: str = SOURCE_CHECKED


#: The licence statement on every UCI dataset page read on :data:`SOURCE_CHECKED`, verbatim.
UCI_CC_BY_4 = (
    "This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) "
    "license. This allows for the sharing and adaptation of the datasets for any purpose, "
    "provided that the appropriate credit is given."
)

#: Delve's copyright notice, which covers kin8nm, pumadyn32nh and comp-activ (cpu_activity).
_DELVE = (
    "Copyright (c) 1995-1996 by The University of Toronto, Toronto, Ontario, Canada. All Rights "
    "Reserved Permission to use, copy, modify, and distribute this software and its "
    "documentation for non-commercial purposes only is hereby granted without fee"
)


def _uci(uci_id: int, slug: str, note: str = "") -> SourceLicence:
    """An included entry for a dataset whose UCI page states :data:`UCI_CC_BY_4`."""
    reason = "UCI page states CC BY 4.0" + (f"; {note}" if note else "")
    return SourceLicence(f"https://archive.ics.uci.edu/dataset/{uci_id}/{slug}", UCI_CC_BY_4,
                         "CC BY 4.0", True, reason)


def _none(url: str, reason: str) -> SourceLicence:
    """An excluded entry for a source that states no licence or could not be read."""
    return SourceLicence(url, "", "", False, reason)


def _kaggle(ref: str, include: bool, reason: str) -> SourceLicence:
    """An entry read from a Kaggle dataset's licence field (``licenseNameNullable`` from
    ``https://www.kaggle.com/api/v1/datasets/view/<ref>``, the field the page displays)."""
    return SourceLicence(f"https://www.kaggle.com/datasets/{ref}", "CC0: Public Domain",
                         "CC0: Public Domain", include, reason)


_MFEAT = "one UCI page (Multiple Features) for all six mfeat sets"
_PROMISE = ("tera-PROMISE page states no licence and links no data DOI; no Zenodo record found "
            "under its name; NASA MDP defect data, origin licence unverified")
_ELENA = ("KEEL page says the data was obtained from the ELENA project and states no licence (only "
          "KEEL's site copyright); the ELENA page timed out")

#: Licence at the original source for every candidate that passes the overlap filters, keyed
#: by OpenML dataset id and read on :data:`SOURCE_CHECKED` from the page the OpenML
#: description names (traced to its current address where the old one moved). Only page text
#: and metadata were read; no dataset contents were downloaded for this table.
SOURCE_LICENCES: dict[int, SourceLicence] = {
    # OpenML-CC18, UCI pages.
    3: _uci(22, "chess+king+rook+vs+king+pawn"),
    6: _uci(59, "letter+recognition"),
    11: _uci(12, "balance+scale"),
    12: _uci(72, "multiple+features", _MFEAT),
    14: _uci(72, "multiple+features", _MFEAT),
    15: _uci(15, "breast+cancer+wisconsin+original"),
    16: _uci(72, "multiple+features", _MFEAT),
    18: _uci(72, "multiple+features", _MFEAT),
    22: _uci(72, "multiple+features", _MFEAT),
    23: _uci(30, "contraceptive+method+choice"),
    28: _uci(80, "optical+recognition+of+handwritten+digits"),
    32: _uci(81, "pen+based+recognition+of+handwritten+digits",
             "OpenML's original_data_url names optdigits; its description names this page"),
    44: _uci(94, "spambase"),
    50: _uci(101, "tic+tac+toe+endgame"),
    182: _uci(146, "statlog+landsat+satellite"),
    300: _uci(54, "isolet"),
    307: _uci(152, "connectionist+bench+vowel+recognition+deterding+data"),
    1462: _uci(267, "banknote+authentication"),
    1468: _uci(233, "cnae+9"),
    1475: _uci(249, "first+order+theorem+proving"),
    1478: _uci(240, "human+activity+recognition+using+smartphones"),
    1485: _uci(171, "madelon"),
    1486: _uci(227, "nomao"),
    1487: _uci(172, "ozone+level+detection"),
    1497: _uci(194, "wall+following+robot+navigation+data"),
    1501: _uci(178, "semeion+handwritten+digit"),
    1510: _uci(17, "breast+cancer+wisconsin+diagnostic"),
    1590: _uci(2, "adult"),
    4534: _uci(327, "phishing+websites"),
    4538: _uci(302, "gesture+phase+segmentation"),
    6332: _uci(32, "cylinder+bands"),
    23381: _uci(289, "dresses+attribute+sales"),
    40668: _uci(26, "connect+4"),
    40923: _uci(389, "devanagari+handwritten+character+dataset",
                "UCI lists the creators (Shailesh Acharya, Prashnna Gyawali); OpenML cites "
                "their archived cvresearchnepal page and a Kaggle mirror"),
    40975: _uci(19, "car+evaluation"),
    40978: _uci(51, "internet+advertisements"),
    40979: _uci(72, "multiple+features", _MFEAT),
    40982: _uci(198, "steel+plates+faults"),
    40983: _uci(285, "wilt"),
    40984: _uci(50, "image+segmentation"),
    40994: _uci(252, "climate+model+simulation+crashes"),
    # OpenML-CC18, other sources.
    151: _none("http://www.inescporto.pt/~jgama/ales/ales_5.html",
               "source page returns 404; MOA's datasets page (https://moa.cms.waikato.ac.nz/"
               "datasets/) describes Elec2 without a licence; unverifiable"),
    188: _none("https://ml.cms.waikato.ac.nz/weka/datasets.html",
               "Weka datasets page (moved from www.cs.waikato.ac.nz) states no licence for the "
               "collection; unverifiable"),
    458: _none("https://pages.stern.nyu.edu/~jsimonof/AnalCatData/",
               "book data site (Simonoff, Analyzing Categorical Data) states no licence"),
    469: _none("https://pages.stern.nyu.edu/~jsimonof/AnalCatData/",
               "book data site (Simonoff, Analyzing Categorical Data) states no licence"),
    554: _none("http://yann.lecun.com/exdb/mnist/",
               "the page is an empty directory listing and states no licence; unverifiable"),
    1049: _none("https://openscience.us/repo/defect/mccabehalsted/pc4.html", _PROMISE),
    1050: _none("https://openscience.us/repo/defect/mccabehalsted/pc3.html", _PROMISE),
    1063: _none("https://openscience.us/repo/defect/mccabehalsted/kc2.html", _PROMISE),
    1067: _none("https://openscience.us/repo/defect/mccabehalsted/kc1.html",
                "the page's data DOI (zenodo.268441, CC BY 4.0) is the class-level KC1 with 94 "
                "variables, not this method-level kc1 (21 features); no licence for this one"),
    1068: SourceLicence("https://zenodo.org/records/268456", "cc-by-4.0", "CC BY 4.0", True,
                       "Zenodo record 'pc1' (creator Tim Menzies, the donor), linked as the data "
                       "DOI from https://openscience.us/repo/defect/mccabehalsted/pc1.html; "
                       "licence id read from https://zenodo.org/api/records/268456"),
    1489: _none("https://sci2s.ugr.es/keel/dataset.php?cod=105", _ELENA),
    23517: _kaggle("numerai/encrypted-stock-market-data-from-numerai", True,
                   "Kaggle states CC0; the uploader is Numerai, the data's owner"),
    40499: _none("https://sci2s.ugr.es/keel/dataset.php?cod=72", _ELENA),
    40927: _none("https://www.cs.toronto.edu/~kriz/cifar.html",
                 "the page states no licence (a labelled subset of 80 Million Tiny Images); "
                 "unverifiable"),
    40996: SourceLicence(
        "https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE",
        "The MIT License (MIT) Copyright © 2017 Zalando SE, https://tech.zalando.com Permission "
        "is hereby granted, free of charge, to any person obtaining a copy of this software and "
        "associated documentation files (the “Software”), to deal in the Software without "
        "restriction, including without limitation the rights to use, copy, modify, merge, "
        "publish, distribute, sublicense, and/or sell copies of the Software",
        "MIT", True,
        "the creator's repository, which holds the data, is MIT (README section License "
        "repeats it); MIT grants commercial use and redistribution",
    ),
    41027: _none("https://arxiv.org/abs/1604.07312",
                 "OpenML cites the paper (van Rijn and Vis); arXiv's licence covers the paper, "
                 "and no licence for the generated data is stated"),
    # OpenML-CTR23.
    41021: _kaggle("wduckett/moneyball-mlb-stats-19622012", False,
                   "Kaggle states CC0, but the page says 'The data itself is gathered from "
                   "baseball-reference.com', whose terms (sports-reference.com) returned 403; "
                   "the uploader's right to grant CC0 is unverified"),
    44956: _uci(1, "abalone"),
    44958: _uci(713, "auction+verification"),
    44960: _uci(242, "energy+efficiency"),
    44965: _uci(315, "geographical+original+of+music"),
    44966: _uci(89, "solar+flare"),
    44969: _uci(316, "condition+based+maintenance+of+naval+propulsion+plants"),
    44973: _uci(471, "electrical+grid+stability+simulated+data"),
    44975: _uci(494, "wave+energy+converters"),
    44976: _none("http://gaussianprocess.org/gpml/data/",
                 "GPML data page states no licence; unverifiable"),
    44978: SourceLicence("https://www.cs.toronto.edu/~delve/copyright.html", _DELVE, "", False,
                         "Torgo's page (https://www.dcc.fc.up.pt/~ltorgo/Regression/DataSets.html)"
                         " timed out; the data is Delve's comp-activ, and Delve's notice "
                         "grants non-commercial use only"),
    44980: SourceLicence("https://www.cs.toronto.edu/~delve/copyright.html", _DELVE, "", False,
                         "Delve (data/kin/desc.html) grants non-commercial use only"),
    44981: SourceLicence("https://www.cs.toronto.edu/~delve/copyright.html", _DELVE, "", False,
                         "Delve (data/pumadyn/desc.html) grants non-commercial use only"),
    44984: _none("http://www.personal.psu.edu/hxb11/MEDIAN.HTM",
                 "page timed out (http and https); the data is a sample of the 1988 March CPS, "
                 "a US Census Bureau survey, but the extract's terms could not be read"),
    44987: SourceLicence("https://www.openml.org/api/v1/json/data/44987",
                         "Non-commercial scholarly and teaching purposes.", "", False,
                         "non-commercial; source StatLib (http://lib.stat.cmu.edu/datasets/"
                         "socmob), Biblarz and Raftery (1993)"),
    44989: _kaggle("harlfoxem/housesalesprediction", True,
                   "Kaggle states CC0; the page names no upstream source for the King County "
                   "sales records"),
    44993: SourceLicence("https://cran.r-project.org/package=Ecdat",
                         "License: GPL-2 | GPL-3 [expanded from: GPL (≥ 2)]", "GPL (>= 2)",
                         False,
                         "the R package's copyleft software licence, not a data licence; the "
                         "data's source (Olson 1998, Journal of Applied Econometrics data "
                         "archive, per https://rdrr.io/cran/Ecdat/man/HI.html) states none"),
    44994: _none("https://search.r-project.org/CRAN/refmans/caret/html/cars.html",
                 "caret (GPL-2 | GPL-3) cites Kuiper (2008), Journal of Statistics Education; "
                 "no data licence stated"),
    45012: _kaggle("stefanoleone992/fifa-22-complete-player-dataset", False,
                   "Kaggle states CC0, but the page says 'Data has been scraped from the "
                   "publicly available website sofifa.com', whose terms returned 403; the "
                   "uploader's right to grant CC0 is unverified"),
    45402: _none("http://lib.stat.cmu.edu/datasets/space_ga",
                 "the StatLib file header (Pace and Barry 1997) states no licence"),
}


def licence_verdict(licence: str) -> str | None:
    """Why a recorded licence fails the filter, or ``None`` when it passes.

    Passes CC BY (any version, no NC/ND/SA terms), CC0, an explicit public-domain statement and
    MIT, which grants commercial use and redistribution in terms.

    Args:
        licence: A licence name, as OpenML's field or a :class:`SourceLicence` records it.

    Returns:
        ``None`` for an accepted licence, otherwise the exclusion reason.
    """
    raw = licence.strip()
    norm = "".join(ch for ch in raw.lower() if ch.isalnum())
    if norm == "public":
        return "licence 'Public' (OpenML's upload default; research-use grant only)"
    if not norm:
        return "no licence recorded"
    if norm.startswith("cc0") or "publicdomain" in norm or norm == "mit":
        return None
    if norm.startswith("ccby") and not any(t in norm[4:] for t in ("nc", "nd", "sa")):
        return None
    return f"licence {raw!r} is not CC BY, CC0, public domain or MIT"


def overlap_reasons(candidate: Candidate) -> list[str]:
    """Reasons from the overlap filters alone (TabArena id, name, same source).

    Args:
        candidate: The task to check.

    Returns:
        Reasons in filter order; empty when the candidate shares nothing with either arena.
    """
    reasons = []
    if candidate.dataset_id in {i for i, _ in TABARENA_V0_1}:
        reasons.append("TabArena-v0.1 dataset id")
    reasons += [f"name matches {hit}" for hit in name_clashes(candidate.name)]
    if candidate.dataset_id in SAME_SOURCE:
        reasons.append(f"same source as {SAME_SOURCE[candidate.dataset_id]}")
    return reasons


def source_licence_reason(candidate: Candidate) -> str | None:
    """Why the candidate's source licence keeps it out, or ``None`` when it admits it.

    Args:
        candidate: The task to check.

    Returns:
        ``None`` only for an entry in :data:`SOURCE_LICENCES` marked include whose licence
        passes :func:`licence_verdict`; otherwise the reason, prefixed ``source licence``.
    """
    entry = SOURCE_LICENCES.get(candidate.dataset_id)
    if entry is None:
        return "source licence not verified"
    if not entry.include:
        return f"source licence: {entry.reason}"
    verdict = licence_verdict(entry.licence)
    return f"source licence: {verdict}" if verdict else None


def exclusion_reasons(candidate: Candidate) -> list[str]:
    """Every reason a candidate is kept out of the suite, empty when it is eligible.

    A candidate already excluded by overlap is not checked at source, so its list carries the
    overlap reasons only.

    Args:
        candidate: The task to check.

    Returns:
        Reasons in filter order (TabArena id, name, same source, then source licence).
    """
    reasons = overlap_reasons(candidate)
    if reasons:
        return reasons
    verdict = source_licence_reason(candidate)
    return [verdict] if verdict else []


def eligible(candidates: Iterable[Candidate] = CANDIDATES) -> list[Candidate]:
    """Candidates with no exclusion reason, in task-id order."""
    return sorted((c for c in candidates if not exclusion_reasons(c)), key=lambda c: c.task_id)


def select_suite(
    candidates: Iterable[Candidate] = CANDIDATES,
    targets: dict[str, int] | None = None,
) -> list[Candidate]:
    """The proxy suite: up to the target count of each kind, lowest task ids first.

    Lowest-id-first is arbitrary but fixed, so the suite does not move when it is rebuilt and
    no choice is made by looking at scores.

    Args:
        candidates: The pool to choose from.
        targets: Tasks wanted per kind; defaults to :data:`TARGET_COUNTS`.

    Returns:
        The chosen tasks, grouped by kind in :data:`KINDS` order.
    """
    targets = TARGET_COUNTS if targets is None else targets
    pool = eligible(candidates)
    return [c for kind in KINDS for c in [c for c in pool if c.kind == kind][: targets.get(kind, 0)]]


#: The proxy suite as currently selected.
SUITE: tuple[Candidate, ...] = tuple(select_suite())


def fetch_licence(dataset_id: int, timeout: float = 60.0) -> str:
    """Read one dataset's ``licence`` field from the OpenML API (metadata only, no data file).

    Falls back to ``curl`` when the standard library cannot verify the TLS chain, for the
    reason ``fintfm.evaluation.datasets._download`` gives: the python.org macOS build ships
    without root certificates, and a fallback keeps the dependency list unchanged.

    Args:
        dataset_id: OpenML dataset id.
        timeout: Seconds before the request is abandoned.

    Returns:
        The field verbatim, or an empty string when OpenML records none.

    Raises:
        RuntimeError: If neither urllib nor curl can retrieve the record.
    """
    url = f"https://www.openml.org/api/v1/json/data/{dataset_id}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read()
    except (URLError, OSError) as exc:
        curl = shutil.which("curl")
        if curl is None:
            raise RuntimeError(f"could not fetch {url}: {exc}; curl unavailable") from exc
        result = subprocess.run(
            [curl, "-sSfL", "--max-time", str(int(timeout)), url],
            capture_output=True, check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"could not fetch {url}: {result.stderr.decode().strip()}") from exc
        body = result.stdout
    return json.loads(body)["data_set_description"].get("licence") or ""


def _dry_run(out: TextIO | None = None) -> None:
    """Print the suite, every exclusion with its reasons, and the counts against the target."""
    out = sys.stdout if out is None else out
    suite_ids = {c.task_id for c in SUITE}
    within = sum(c.n_features <= FEATURE_CAP for c in SUITE)
    print(f"Proxy suite (DR4.2): {len(SUITE)} tasks, {within} within the {FEATURE_CAP}-feature "
          f"cap; OpenML metadata fetched {FETCHED}", file=out)
    print(f"licences read at each dataset's original source on {SOURCE_CHECKED} "
          f"(SOURCE_LICENCES); OpenML's field ({LICENCE_SOURCE}) is recorded, not used", file=out)
    pool = eligible()
    for kind in KINDS:
        chosen = [c for c in SUITE if c.kind == kind]
        capped = sum(c.n_features <= FEATURE_CAP for c in chosen)
        n_pool = sum(c.kind == kind for c in pool)
        print(f"\n{kind}: {len(chosen)} of target {TARGET_COUNTS[kind]}, {capped} within the "
              f"{FEATURE_CAP}-feature cap ({n_pool} eligible)", file=out)
        for c in chosen:
            cap = "" if c.n_features <= FEATURE_CAP else f"  (over the {FEATURE_CAP}-feature cap)"
            src = SOURCE_LICENCES[c.dataset_id]
            print(f"  task {c.task_id:>6}  data {c.dataset_id:>5}  {c.name:<32} {c.n_rows:>6} rows "
                  f"x {c.n_features:>4} features  {src.licence} at {src.url}{cap}", file=out)
    excluded = [c for c in CANDIDATES if c.task_id not in suite_ids]
    print(f"\nexcluded: {len(excluded)} of {len(CANDIDATES)} candidates", file=out)
    for c in sorted(excluded, key=lambda c: (c.suite, c.task_id)):
        reasons = exclusion_reasons(c) or ["eligible, beyond the target count"]
        print(f"  {c.suite:<5} task {c.task_id:>6}  data {c.dataset_id:>5}  {c.name:<34} "
              f"{c.kind:<10} {c.n_rows:>6} x {c.n_features:<4} | {'; '.join(reasons)}", file=out)


def _verify_licences(candidates: Sequence[Candidate], out: TextIO | None = None) -> int:
    """Re-fetch each candidate's licence and report changes; returns the number changed."""
    out = sys.stdout if out is None else out
    changed = 0
    for c in candidates:
        live = fetch_licence(c.dataset_id)
        if live != c.licence:
            changed += 1
            print(f"CHANGED data {c.dataset_id} {c.name}: recorded {c.licence!r}, now {live!r}",
                  file=out)
    print(f"{changed} of {len(candidates)} licences changed since {FETCHED}", file=out)
    return changed


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; see the module docstring for the options' purpose."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="print the suite, the exclusions and their reasons")
    parser.add_argument("--task-ids", action="store_true",
                        help="print the suite's OpenML task ids, comma-separated, for the runner")
    parser.add_argument("--verify-licences", action="store_true",
                        help="re-fetch every candidate's OpenML licence field (network)")
    args = parser.parse_args(argv)
    if args.task_ids:
        print(",".join(str(c.task_id) for c in SUITE))
    if args.verify_licences:
        return 1 if _verify_licences(CANDIDATES) else 0
    if args.dry_run or not (args.task_ids or args.verify_licences):
        _dry_run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
