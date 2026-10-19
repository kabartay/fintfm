"""A proxy Elo suite outside TabArena and BeyondArena (design review 2026-10-16, task DR4.2).

TabArena is the benchmark this project reports on, so choosing arms on it biases the number it
reports. The proxy is a second set of OpenML tasks, disjoint from TabArena-v0.1 and BeyondArena,
on which arms can be ranked before the TabArena run that reports them. It is a decision tool
only after its Elo has been shown to rank checkpoints the way TabArena-Lite does (DR4.2's
Verify: Spearman 0.8 or higher across six or more existing checkpoints). Until then, nothing is
decided on it.

**What this module is.** The candidate pool (every task of OpenML-CC18, study 99, and
OpenML-CTR23, study 353), the filters that remove candidates, and the selection that turns what
is left into the suite. ``--dry-run`` prints all of it without fetching anything. The runner
(``run``, ``compare``, ``--row-hash``; see *The runner* below) fetches and scores.

**The filters, in the order the dry run reports them:**

1. *TabArena-v0.1 dataset id* (`fintfm.evaluation.overlap.TABARENA_V0_1`). Catches nothing
   today: TabArena's ids are its own re-uploads (46904 onwards), not these suites' originals.
2. *Name* against TabArena-v0.1 and BeyondArena (`fintfm.evaluation.overlap.name_clashes`).
   Containment over-matches on purpose: ``brazilian_houses`` (Brazilian rentals) is excluded
   for containing TabArena's ``houses`` (California housing), and ``vehicle`` for BeyondArena's
   ``in_vehicle_coupon_recommendation``. A missed overlap costs more than a lost task.
3. *Same source under another name* (:data:`SAME_SOURCE`), found by comparing row and feature
   counts and source URLs against the fork's TabArena and BeyondArena metadata. The contents
   check, ``--row-hash`` (:func:`fintfm.evaluation.overlap.content_fingerprint`), fetches each
   suite dataset and compares it with the TabArena-v0.1 datasets in the local OpenML cache; it
   is a check to run, not a filter applied at import.
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

**The runner (DR4.2).** Written here rather than through TabArena's custom-task pipeline, so it
runs in this repository's environment and nothing is copied from TabArena::

    uv run --extra bench python -m fintfm.evaluation.proxy_suite run \\
        --model runs/x.pt --out runs/proxy/x.csv
    uv run --extra bench python -m fintfm.evaluation.proxy_suite compare \\
        runs/proxy/x.csv runs/proxy/reference.csv
    uv run --extra bench python -m fintfm.evaluation.proxy_suite --row-hash

* *Data and splits.* Each task's definition (``/task/<id>``), its split file and its dataset
  are fetched from OpenML and cached under :data:`PROXY_CACHE` (``data/cache``, gitignored).
  The splits are **OpenML's own**: 34 of the 35 suite tasks define 10-fold cross-validation
  with one repeat, and energy_efficiency (361617) 10 repeats of 10 folds (read from the task
  definitions on 2026-10-19; each run records every task's procedure in its ``.json``). The
  runner scores repeat 0, folds 0 to 2 of each (:data:`N_SPLITS` = 3), so three splits per
  task whatever the procedure. Every method sees the same training and test rows of a split.
* *Methods.* FinTFM from a checkpoint and a configuration's ``inference`` section
  (:meth:`FinancialTFMClassifier.from_config`; for regression the same section passed to
  :class:`FinancialTFMRegressor`), skipped with a recorded reason when a task is wider than
  the checkpoint's ``max_features`` or has more classes than its head. Baselines
  (:data:`BASELINES`, scikit-learn and LightGBM defaults, LightGBM only when installed) are
  cached per task and library version under :data:`PROXY_CACHE`, so a second checkpoint reuses
  them.
* *Scores.* One row per task, split and method in a CSV (:data:`RESULT_COLUMNS`). The error
  compared is 1 - ROC-AUC (binary), log loss (multiclass) or RMSE (regression); binary rows also
  carry the minority class's average precision.
* *Elo* (:func:`elo_table`): a Bradley-Terry model fitted to every pairwise result on every
  split (lower error wins, an equal error is half a win each), on the Elo scale
  (400 log10 of strength), anchored at :data:`ANCHOR` = 1000, with a 95% percentile interval
  from resampling tasks with replacement. Reimplemented from the model's definition (Hunter's
  MM algorithm); it is not TabArena's code and its numbers are not on TabArena's scale, so
  only ranks compare across the two.
* *Sign test* (:func:`sign_test`): wins and losses of a checkpoint against a reference
  checkpoint over tasks (each task's mean error over its splits; ties dropped), two-sided
  exact binomial. Splits of one task share most of their rows, so ``--unit split`` is reported
  as secondary only.

**Evaluation only.** No OpenML data trains anything here; ``CLAUDE.md``'s licensing boundary
bans third-party training data and permits evaluation.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO
from urllib.error import URLError

import numpy as np
import pandas as pd

from fintfm.evaluation.datasets import CACHE_DIR
from fintfm.evaluation.overlap import (
    REPORTING_FLAG,
    TABARENA_OPENML_CACHE,
    TABARENA_V0_1,
    guard_selection,
    load_reference_fingerprints,
    name_clashes,
    row_hash_report,
)

if TYPE_CHECKING:
    from fintfm.config import Config
    from fintfm.modeling.model import FinancialTFM

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
    body = _http_get(f"https://www.openml.org/api/v1/json/data/{dataset_id}", timeout)
    return json.loads(body)["data_set_description"].get("licence") or ""


def _http_get(url: str, timeout: float = 60.0) -> bytes:
    """GET a URL with urllib, falling back to ``curl`` (see :func:`fetch_licence` for why).

    Raises:
        RuntimeError: If neither urllib nor curl can retrieve it.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read()
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
        return result.stdout


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


# --------------------------------------------------------------------------------------------
# The runner (DR4.2): fetch, score, Elo, sign test. See "The runner" in the module docstring.
# --------------------------------------------------------------------------------------------

#: OpenML task definitions, split files, datasets and cached baseline scores (gitignored).
PROXY_CACHE = CACHE_DIR / "proxy_suite"

#: Splits scored per task: repeat 0, folds 0 to ``N_SPLITS - 1`` of the task's own OpenML
#: cross-validation (10 folds for every suite task; one task repeats it 10 times).
N_SPLITS = 3

#: The method Elo is anchored at, and its rating. TabArena's anchor is "RF (default)"; this is
#: scikit-learn's RandomForest at its defaults.
ANCHOR = "RF"
ANCHOR_RATING = 1000.0

#: Baselines scored on every split, at their libraries' defaults. LightGBM only if installed.
BASELINES: tuple[str, ...] = ("RF", "ExtraTrees", "Linear", "KNN", "LightGBM")

#: One row per task, split and method. ``error`` is what Elo and the sign test compare (lower
#: is better); ``status`` is ``ok`` or the reason a method has no score on that split.
RESULT_COLUMNS: tuple[str, ...] = (
    "task_id", "dataset_id", "name", "kind", "split", "repeat", "fold", "method", "metric",
    "error", "roc_auc", "ap", "n_train", "n_test", "n_features", "seconds", "status",
)

#: Error metric per kind.
METRICS: dict[str, str] = {"binary": "1-roc_auc", "multiclass": "log_loss", "regression": "rmse"}


@dataclass
class ProxyTask:
    """One fetched suite task, ready to score.

    Attributes:
        candidate: The suite entry.
        X: Features, categorical columns as strings (NaN kept), numeric columns as floats.
        y: Integer class codes ``0..K-1`` (classification) or floats (regression).
        splits: ``(repeat, fold, train_rows, test_rows)`` from OpenML's split file, in order.
        procedure: The task's estimation procedure as OpenML states it.
    """

    candidate: Candidate
    X: pd.DataFrame
    y: np.ndarray
    splits: list[tuple[int, int, np.ndarray, np.ndarray]]
    procedure: str


def ensure_ca_bundle() -> None:
    """Point ``SSL_CERT_FILE`` at a CA bundle when none is set, for ``fetch_openml``.

    ``fetch_openml`` owns its transport, so the curl fallback cannot reach it; the python.org
    and uv macOS builds verify no TLS chain without a bundle. certifi when installed, else the
    system's ``/etc/ssl/cert.pem``.
    """
    if "SSL_CERT_FILE" in os.environ:
        return
    try:
        import certifi

        os.environ["SSL_CERT_FILE"] = certifi.where()
    except ImportError:
        if Path("/etc/ssl/cert.pem").exists():
            os.environ["SSL_CERT_FILE"] = "/etc/ssl/cert.pem"


def _cached_get(url: str, path: Path) -> bytes:
    """``url``'s body, read from ``path`` when cached and written there when not."""
    if path.exists():
        return path.read_bytes()
    body = _http_get(url, timeout=300.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(body)
    tmp.replace(path)
    return body


def task_definition(task_id: int, cache_dir: Path = PROXY_CACHE) -> dict[str, Any]:
    """An OpenML task's dataset, target, split URL and estimation procedure (cached).

    Args:
        task_id: OpenML task id.
        cache_dir: Runner cache root.

    Returns:
        ``dataset_id``, ``target``, ``splits_url``, ``repeats``, ``folds`` and ``procedure``.
    """
    url = f"https://www.openml.org/api/v1/json/task/{task_id}"
    task = json.loads(_cached_get(url, cache_dir / "tasks" / f"{task_id}.json"))["task"]
    inputs = {i["name"]: i for i in task["input"]}
    data = inputs["source_data"]["data_set"]
    est = inputs["estimation_procedure"]["estimation_procedure"]
    params = {p["name"]: p.get("value") for p in est.get("parameter", [])}
    repeats, folds = int(params.get("number_repeats") or 1), int(params.get("number_folds") or 1)
    return {
        "dataset_id": int(data["data_set_id"]),
        "target": data["target_feature"],
        "splits_url": est["data_splits_url"],
        "repeats": repeats,
        "folds": folds,
        "procedure": f"{est['type']}, {repeats} repeat(s) x {folds} fold(s)",
    }


def parse_splits_arff(text: str) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """Parse an OpenML split file (``type``, ``rowid``, ``repeat``, ``fold`` columns).

    Args:
        text: The ARFF file's text.

    Returns:
        ``(repeat, fold)`` to ``(train_rows, test_rows)``, each sorted. Learning-curve
        ``sample`` columns, absent from these suites' tasks, are ignored.

    Raises:
        ValueError: If a required column is missing.
    """
    attrs: list[str] = []
    rows: list[list[str]] = []
    in_data = False
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("%"):
            continue
        if not in_data:
            low = s.lower()
            if low.startswith("@attribute"):
                attrs.append(s.split()[1].strip("'\"").lower())
            elif low.startswith("@data"):
                in_data = True
            continue
        rows.append([v.strip() for v in s.split(",")])
    try:
        i_type, i_row, i_rep, i_fold = (attrs.index(a) for a in ("type", "rowid", "repeat", "fold"))
    except ValueError as exc:
        raise ValueError(f"split file lacks a required column; has {attrs}") from exc
    buckets: dict[tuple[int, int], tuple[list[int], list[int]]] = {}
    for r in rows:
        key = (int(float(r[i_rep])), int(float(r[i_fold])))
        train, test = buckets.setdefault(key, ([], []))
        (train if r[i_type].strip("'\"").upper() == "TRAIN" else test).append(int(float(r[i_row])))
    return {k: (np.array(sorted(tr)), np.array(sorted(te))) for k, (tr, te) in buckets.items()}


def openml_frame(
    dataset_id: int, target: str | None, data_home: Path | None
) -> tuple[pd.DataFrame, pd.Series | None]:
    """One OpenML dataset's features and target, through scikit-learn's cached fetcher.

    Args:
        dataset_id: OpenML dataset id.
        target: Target column; None for the dataset's default target.
        data_home: scikit-learn's cache root; None for its default (``~/scikit_learn_data``).

    Returns:
        ``(features, target)``, rows in OpenML's order (which split files index).
    """
    from sklearn.datasets import fetch_openml

    ensure_ca_bundle()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bunch = fetch_openml(
            data_id=dataset_id, target_column=target or "default-target", as_frame=True,
            parser="auto", data_home=None if data_home is None else str(data_home),
        )
    return bunch.data, bunch.target


def _prepare(X: pd.DataFrame) -> pd.DataFrame:
    """Categorical, object and bool columns as strings (NaN kept), the rest as floats."""
    out = {}
    for name in X.columns:
        col = X[name]
        if pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col):
            out[name] = col.astype(float)
        else:
            out[name] = col.astype("object").map(lambda v: v if pd.isna(v) else str(v))
    return pd.DataFrame(out, index=X.index)


def fetch_task(candidate: Candidate, cache_dir: Path = PROXY_CACHE,
               n_splits: int = N_SPLITS) -> ProxyTask:
    """Fetch a suite task's data and OpenML splits (network on first use, cached after).

    Args:
        candidate: The suite entry.
        cache_dir: Runner cache root.
        n_splits: Folds of repeat 0 to keep.

    Returns:
        The :class:`ProxyTask`.

    Raises:
        RuntimeError: If OpenML's task names another dataset than the table records, or
            defines fewer than ``n_splits`` folds.
    """
    ensure_ca_bundle()
    task = task_definition(candidate.task_id, cache_dir)
    if task["dataset_id"] != candidate.dataset_id:
        raise RuntimeError(f"task {candidate.task_id} is on dataset {task['dataset_id']}, "
                           f"the table records {candidate.dataset_id}")
    text = _cached_get(task["splits_url"],
                       cache_dir / "tasks" / f"{candidate.task_id}_splits.arff").decode()
    all_splits = parse_splits_arff(text)
    keys = [(0, f) for f in range(n_splits)]
    missing = [k for k in keys if k not in all_splits]
    if missing:
        raise RuntimeError(f"task {candidate.task_id} ({task['procedure']}) has no split {missing}")
    X, y = openml_frame(candidate.dataset_id, task["target"], cache_dir / "openml")
    if candidate.kind == "regression":
        target = np.asarray(y, dtype=float)
    else:
        labels = np.asarray(y).astype(str)
        target = np.unique(labels, return_inverse=True)[1].astype(np.int64)
    splits = [(r, f, *all_splits[(r, f)]) for r, f in keys]
    return ProxyTask(candidate, _prepare(X), target, splits, task["procedure"])


def _columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """``(numeric, categorical)`` column names of a :func:`_prepare`-d frame."""
    num = [c for c in X.columns if pd.api.types.is_float_dtype(X[c])]
    return num, [c for c in X.columns if c not in set(num)]


def _baseline(method: str, kind: str, X: pd.DataFrame, seed: int, n_jobs: int) -> Any:
    """An unfitted baseline pipeline at its library's defaults.

    Trees and LightGBM get median-imputed numerics and ordinal categories; the linear model
    and KNN get standardised numerics and one-hot categories.

    Raises:
        ImportError: For ``LightGBM`` when it is not installed.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        ExtraTreesRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

    num, cat = _columns(X)
    clf = kind != "regression"
    ordinal = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                               encoded_missing_value=-1), cat),
    ])
    onehot = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    if method == "RF":
        est = (RandomForestClassifier if clf else RandomForestRegressor)(
            random_state=seed, n_jobs=n_jobs)
        return make_pipeline(ordinal, est)
    if method == "ExtraTrees":
        est = (ExtraTreesClassifier if clf else ExtraTreesRegressor)(
            random_state=seed, n_jobs=n_jobs)
        return make_pipeline(ordinal, est)
    if method == "Linear":
        return make_pipeline(onehot, LogisticRegression(max_iter=1000) if clf else Ridge())
    if method == "KNN":
        return make_pipeline(onehot, KNeighborsClassifier() if clf else KNeighborsRegressor())
    if method == "LightGBM":
        import lightgbm

        est = (lightgbm.LGBMClassifier if clf else lightgbm.LGBMRegressor)(
            random_state=seed, n_jobs=n_jobs, verbose=-1)
        return make_pipeline(ordinal, est)
    raise ValueError(f"unknown baseline {method!r}")


def available_baselines() -> tuple[str, ...]:
    """:data:`BASELINES` less LightGBM when it is not installed."""
    try:
        import lightgbm  # noqa: F401
    except (ImportError, OSError):
        # OSError: lightgbm's libomp missing on macOS raises it at import (FINDINGS §2)
        return tuple(m for m in BASELINES if m != "LightGBM")
    return BASELINES


def _fintfm_matrix(X: pd.DataFrame, train: np.ndarray, test: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Float32 train and test matrices for FinTFM: numerics as they are (NaN kept, the model
    reads missingness), categories as ordinal codes fitted on the training rows, unseen or
    missing categories as NaN."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OrdinalEncoder

    num, cat = _columns(X)
    ct = ColumnTransformer([
        ("num", "passthrough", num),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                               encoded_missing_value=np.nan), cat),
    ])
    Xtr = ct.fit_transform(X.iloc[train])
    return Xtr.astype(np.float32), ct.transform(X.iloc[test]).astype(np.float32)


def _fintfm(kind: str, model: FinancialTFM, cfg: Config, seed: int, device: str) -> Any:
    """FinTFM's unfitted estimator, every setting from ``cfg.inference``."""
    from fintfm.inference.binning import DEFAULT_N_BINS
    from fintfm.inference.classifier import FinancialTFMClassifier
    from fintfm.inference.regressor import FinancialTFMRegressor

    if kind != "regression":
        return FinancialTFMClassifier.from_config(
            model, cfg.inference, device=device, random_state=seed)
    params = {f.name: getattr(cfg.inference, f.name) for f in fields(cfg.inference)}
    return FinancialTFMRegressor(
        model, n_bins=min(DEFAULT_N_BINS, model.cfg.max_classes),
        classifier_params={**params, "device": device, "random_state": seed},
    )


def _score(kind: str, y: np.ndarray, est: Any, X_test: Any, n_classes: int
           ) -> dict[str, float]:
    """``error``, ``roc_auc`` and ``ap`` for one fitted estimator on one test split."""
    from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

    nan = float("nan")
    if kind == "regression":
        pred = np.asarray(est.predict(X_test), dtype=float)
        return {"error": float(np.sqrt(np.mean((pred - y) ** 2))), "roc_auc": nan, "ap": nan}
    proba = np.asarray(est.predict_proba(X_test), dtype=float)
    full = np.zeros((len(y), n_classes))
    full[:, np.asarray(est.classes_, dtype=int)] = proba
    full = np.clip(full, 1e-15, None)
    full /= full.sum(axis=1, keepdims=True)
    if kind == "multiclass":
        return {"error": float(log_loss(y, full, labels=range(n_classes))), "roc_auc": nan,
                "ap": nan}
    if len(np.unique(y)) < 2:
        return {"error": nan, "roc_auc": nan, "ap": nan}
    auc = float(roc_auc_score(y, full[:, 1]))
    minority = int(np.argmin(np.bincount(y, minlength=2)))
    ap = float(average_precision_score(y == minority, full[:, minority]))
    return {"error": 1.0 - auc, "roc_auc": auc, "ap": ap}


def _row(task: ProxyTask, split: int, method: str, n_features: int, **values: Any
         ) -> dict[str, Any]:
    """One :data:`RESULT_COLUMNS` row."""
    c = task.candidate
    rep, fold, train, test = task.splits[split]
    base = {"task_id": c.task_id, "dataset_id": c.dataset_id, "name": c.name, "kind": c.kind,
            "split": split, "repeat": rep, "fold": fold, "method": method,
            "metric": METRICS[c.kind], "error": float("nan"), "roc_auc": float("nan"),
            "ap": float("nan"), "n_train": len(train), "n_test": len(test),
            "n_features": n_features, "seconds": float("nan"), "status": "ok"}
    return {**base, **values}


def _fit_and_score(task: ProxyTask, split: int, method: str, est: Any, X_train: Any,
                   X_test: Any, n_features: int) -> dict[str, Any]:
    """Fit on a split's training rows, score on its test rows; a fit-time refusal (a task
    wider than the checkpoint, more classes than its head) is recorded, not raised."""
    _, _, train, test = task.splits[split]
    n_classes = int(task.y.max()) + 1 if task.candidate.kind != "regression" else 0
    t0 = time.time()
    try:
        est.fit(X_train, task.y[train])
        scores = _score(task.candidate.kind, task.y[test], est, X_test, n_classes)
    except ValueError as exc:
        return _row(task, split, method, n_features, status=f"skipped: {exc}".splitlines()[0])
    status = "ok" if np.isfinite(scores["error"]) else "skipped: one class in the test split"
    return _row(task, split, method, n_features, seconds=round(time.time() - t0, 3),
                status=status, **scores)


def _library_tag() -> str:
    """scikit-learn and LightGBM versions, which key the baseline cache."""
    import sklearn

    try:
        import lightgbm

        lgb = lightgbm.__version__
    except (ImportError, OSError):
        lgb = "none"
    return f"sklearn-{sklearn.__version__}_lightgbm-{lgb}"


def baseline_rows(task: ProxyTask, cache_dir: Path = PROXY_CACHE, seed: int = 0,
                  n_jobs: int = 1, log: Callable[[str], None] = print) -> list[dict[str, Any]]:
    """Every baseline's rows on a task's splits, from the cache where present.

    The cache is one CSV per task under ``<cache_dir>/baselines/<library versions>/seed<seed>``;
    missing (split, method) pairs are scored and added, so a run with more splits extends it.

    Returns:
        :data:`RESULT_COLUMNS` rows for every available baseline and split.
    """
    c = task.candidate
    path = cache_dir / "baselines" / _library_tag() / f"seed{seed}" / f"task_{c.task_id}.csv"
    cached = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=RESULT_COLUMNS)
    have = {(int(s), m) for s, m in zip(cached["split"], cached["method"], strict=True)}
    new: list[dict[str, Any]] = []
    for split, (_, fold, train, test) in enumerate(task.splits):
        for method in available_baselines():
            if (split, method) in have:
                continue
            est = _baseline(method, c.kind, task.X, seed, n_jobs)
            row = _fit_and_score(task, split, method, est, task.X.iloc[train],
                                 task.X.iloc[test], task.X.shape[1])
            log(f"  {c.name:<28} fold {fold} {method:<11} {row['metric']} {row['error']:.4f} "
                f"({row['seconds']} s) {row['status'] if row['status'] != 'ok' else ''}")
            new.append(row)
    if new:
        cached = pd.concat([cached, pd.DataFrame(new, columns=RESULT_COLUMNS)], ignore_index=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        cached.to_csv(path, index=False)
    keep = cached["split"].astype(int).lt(len(task.splits))
    keep &= cached["method"].isin(available_baselines())
    return cached[keep].sort_values(["split", "method"]).to_dict("records")


def fintfm_rows(task: ProxyTask, model: FinancialTFM, cfg: Config, method: str, seed: int = 0,
                device: str = "cpu", log: Callable[[str], None] = print) -> list[dict[str, Any]]:
    """FinTFM's rows on a task's splits; a split it cannot take is a ``skipped`` row."""
    rows = []
    for split, (_, fold, train, test) in enumerate(task.splits):
        Xtr, Xte = _fintfm_matrix(task.X, train, test)
        est = _fintfm(task.candidate.kind, model, cfg, seed, device)
        row = _fit_and_score(task, split, method, est, Xtr, Xte, Xtr.shape[1])
        log(f"  {task.candidate.name:<28} fold {fold} {method:<11} {row['metric']} "
            f"{row['error']:.4f} ({row['seconds']} s) "
            f"{row['status'] if row['status'] != 'ok' else ''}")
        rows.append(row)
    return rows


def run_suite(
    model_path: str | None,
    out: Path,
    *,
    tasks: Sequence[Candidate] | None = None,
    cfg: Config | None = None,
    name: str | None = None,
    n_splits: int = N_SPLITS,
    baselines: bool = True,
    cache_dir: Path = PROXY_CACHE,
    n_jobs: int = 1,
    device: str = "cpu",
    seed: int = 0,
    allow_reporting: bool = False,
    log: Callable[[str], None] = print,
) -> pd.DataFrame:
    """Score a checkpoint and the baselines on the suite and write the per-split CSV.

    The CSV is rewritten after each task, so an interrupted run keeps what it finished. A
    ``<out>.json`` beside it records the checkpoint, configuration layers, splits and each
    task's OpenML estimation procedure.

    Args:
        model_path: FinTFM checkpoint; None scores (and caches) the baselines only.
        out: CSV path.
        tasks: Suite entries to score; :data:`SUITE` when None.
        cfg: Configuration whose ``inference`` section builds FinTFM; the packaged default
            when None.
        name: FinTFM's method name; ``FinTFM:<checkpoint stem>`` when None.
        n_splits: Folds of repeat 0 per task.
        baselines: Whether to include baseline rows.
        cache_dir: Runner cache root.
        n_jobs: Threads for the tree baselines and LightGBM.
        device: Torch device for FinTFM.
        seed: Seed for every estimator.
        allow_reporting: :data:`fintfm.evaluation.overlap.REPORTING_FLAG` was passed.
        log: Where progress lines go.

    Returns:
        The rows written.

    Raises:
        SelectionForbiddenError: If a task is a selection-forbidden dataset and
            ``allow_reporting`` is false.
    """
    from fintfm.config import load_config

    tasks = list(SUITE if tasks is None else tasks)
    reporting = guard_selection([t.name for t in tasks], allow_reporting, "proxy_suite run")
    cfg = cfg if cfg is not None else load_config()
    model = None
    if model_path is not None:
        from fintfm.modeling.model import FinancialTFM

        model = FinancialTFM.load(model_path)
        name = name or f"FinTFM:{Path(model_path).stem}"
    out.parent.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {
        "model": model_path, "method": name, "config_sources": list(cfg.sources),
        "n_splits": n_splits, "splits": f"repeat 0, folds 0-{n_splits - 1}", "seed": seed,
        "baselines": list(available_baselines()) if baselines else [],
        "libraries": _library_tag(), "reporting_only": reporting, "procedures": {},
    }
    rows: list[dict[str, Any]] = []
    for cand in tasks:
        task = fetch_task(cand, cache_dir, n_splits)
        meta["procedures"][str(cand.task_id)] = task.procedure
        log(f"task {cand.task_id} {cand.name}: {len(task.y)} rows x {task.X.shape[1]} features, "
            f"{task.procedure}, scoring {n_splits} split(s)")
        if baselines:
            rows += baseline_rows(task, cache_dir, seed, n_jobs, log)
        if model is not None:
            assert name is not None
            rows += fintfm_rows(task, model, cfg, name, seed, device, log)
        pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(out, index=False)
        out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    frame = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    frame.to_csv(out, index=False)
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return frame


def _task_win_matrices(results: pd.DataFrame, methods: Sequence[str]) -> dict[int, np.ndarray]:
    """Per task, ``W[i, j]`` = splits on which method ``i`` beat ``j`` (a tie is half each)."""
    idx = {m: i for i, m in enumerate(methods)}
    ok = results[np.isfinite(results["error"].astype(float)) & results["method"].isin(idx)]
    out: dict[int, np.ndarray] = {}
    for (tid, _split), g in ok.groupby(["task_id", "split"]):
        err = dict(zip(g["method"], g["error"].astype(float), strict=True))
        W = out.setdefault(int(tid), np.zeros((len(methods), len(methods))))
        for a, b in itertools.combinations(sorted(err, key=idx.__getitem__), 2):
            i, j = idx[a], idx[b]
            if err[a] < err[b]:
                W[i, j] += 1.0
            elif err[b] < err[a]:
                W[j, i] += 1.0
            else:
                W[i, j] += 0.5
                W[j, i] += 0.5
    return out


def bradley_terry(W: np.ndarray, prior: float = 0.5, max_iter: int = 10_000,
                  tol: float = 1e-10) -> np.ndarray:
    """Bradley-Terry log-strengths by Hunter's (2004) MM algorithm.

    ``P(i beats j) = p_i / (p_i + p_j)``; the update is ``p_i <- w_i / sum_j n_ij / (p_i +
    p_j)`` with ``w_i`` method ``i``'s wins and ``n_ij`` the pair's battles. Each pair that met
    gets ``prior`` extra wins each way, which keeps an undefeated method's strength finite (a
    bootstrap resample produces one easily) and shrinks thinly measured pairs toward even.

    Args:
        W: ``W[i, j]`` wins of ``i`` over ``j``.
        prior: Pseudo-wins added each way to every pair with at least one battle.
        max_iter: Iteration cap.
        tol: Convergence threshold on the largest log-strength change.

    Returns:
        Natural-log strengths centred at mean zero over methods with battles; NaN for a
        method with none.
    """
    played = (W + W.T) > 0
    W = W + prior * played
    N = W + W.T
    active = played.any(axis=1)
    logp = np.full(len(W), np.nan)
    if not active.any():
        return logp
    Wa, Na = W[np.ix_(active, active)], N[np.ix_(active, active)]
    wins = Wa.sum(axis=1)
    p = np.ones(len(Wa))
    for _ in range(max_iter):
        new = wins / (Na / (p[:, None] + p[None, :])).sum(axis=1)
        new /= np.exp(np.log(new).mean())
        done = np.max(np.abs(np.log(new) - np.log(p))) < tol
        p = new
        if done:
            break
    logp[active] = np.log(p)
    return logp


def elo_table(results: pd.DataFrame, anchor: str = ANCHOR, n_boot: int = 1000, seed: int = 0,
              prior: float = 0.5) -> pd.DataFrame:
    """Elo-scale Bradley-Terry ratings over every pairwise per-split result.

    Args:
        results: :data:`RESULT_COLUMNS` rows, any number of methods.
        anchor: The method rated :data:`ANCHOR_RATING`.
        n_boot: Bootstrap resamples of tasks (0 for none).
        seed: Bootstrap seed.
        prior: :func:`bradley_terry`'s pseudo-wins.

    Returns:
        One row per method, best first: ``elo``, ``lo`` and ``hi`` (2.5th and 97.5th
        percentiles over resamples), ``battles`` and ``tasks`` (with a score).

    Raises:
        ValueError: If ``anchor`` has no scored split.
    """
    ok = results[np.isfinite(results["error"].astype(float))]
    methods = sorted(ok["method"].unique())
    if anchor not in methods:
        raise ValueError(f"anchor {anchor!r} has no scored split; methods: {methods}")
    mats = _task_win_matrices(results, methods)
    if not mats:
        raise ValueError("no split has two scored methods, so there is nothing to rate")
    a = methods.index(anchor)
    scale = 400.0 / np.log(10.0)

    def ratings(W: np.ndarray) -> np.ndarray:
        """Elo ratings for one win matrix, anchored."""
        logp = bradley_terry(W, prior)
        return ANCHOR_RATING + scale * (logp - logp[a])

    total = sum(mats.values())
    point = ratings(total)
    boot = np.full((n_boot, len(methods)), np.nan)
    rng = np.random.default_rng(seed)
    task_ids = list(mats)
    for b in range(n_boot):
        pick = rng.choice(len(task_ids), size=len(task_ids), replace=True)
        boot[b] = ratings(sum(mats[task_ids[k]] for k in pick))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0) if n_boot else (point, point)
    table = pd.DataFrame({
        "method": methods, "elo": point, "lo": lo, "hi": hi,
        "battles": (total + total.T).sum(axis=1).astype(int),
        "tasks": [ok.loc[ok["method"] == m, "task_id"].nunique() for m in methods],
    })
    return table.sort_values("elo", ascending=False, ignore_index=True)


@dataclass(frozen=True)
class SignTest:
    """A paired sign test of one method against another.

    Attributes:
        candidate: The method under test.
        reference: The method it is compared against.
        unit: ``"task"`` (mean error over a task's splits) or ``"split"``.
        wins: Units where the candidate's error is lower.
        losses: Units where it is higher.
        ties: Units with equal error, dropped from the test.
        p_value: Two-sided exact binomial p at 0.5; NaN with no untied unit.
        mean_difference: Mean of reference error minus candidate error (positive favours the
            candidate). Errors are on different scales across kinds, so read the sign only.
    """

    candidate: str
    reference: str
    unit: str
    wins: int
    losses: int
    ties: int
    p_value: float
    mean_difference: float


def sign_test(results: pd.DataFrame, candidate: str, reference: str, unit: str = "task"
              ) -> SignTest:
    """Paired sign test over the tasks (or splits) both methods scored.

    Args:
        results: :data:`RESULT_COLUMNS` rows holding both methods.
        candidate: Method under test.
        reference: Reference method.
        unit: ``"task"`` or ``"split"``.

    Returns:
        The :class:`SignTest`.
    """
    from scipy.stats import binomtest

    if unit not in ("task", "split"):
        raise ValueError(f"unit must be 'task' or 'split', got {unit!r}")
    ok = results[np.isfinite(results["error"].astype(float))]
    key = ["task_id", "split"]

    def errors(method: str) -> pd.DataFrame:
        """One method's error per (task, split)."""
        return ok.loc[ok["method"] == method, [*key, "error"]].astype({"error": float})

    pair = errors(candidate).merge(errors(reference), on=key, suffixes=("_c", "_r"))
    if unit == "task":
        pair = pair.groupby("task_id", as_index=False)[["error_c", "error_r"]].mean()
    diff = (pair["error_r"] - pair["error_c"]).to_numpy()
    wins, losses = int((diff > 0).sum()), int((diff < 0).sum())
    p = binomtest(wins, wins + losses, 0.5).pvalue if wins + losses else float("nan")
    return SignTest(candidate, reference, unit, wins, losses, int((diff == 0).sum()), float(p),
                    float(diff.mean()) if len(diff) else float("nan"))


def _print_elo(table: pd.DataFrame, anchor: str, n_boot: int) -> None:
    """Print an :func:`elo_table`."""
    print(f"\nElo (Bradley-Terry over per-split pairwise results, {anchor} = "
          f"{ANCHOR_RATING:.0f}; 95% interval from {n_boot} task resamples)")
    print(f"{'method':<32}{'elo':>8}{'95% interval':>20}{'battles':>9}{'tasks':>7}")
    for r in table.itertuples():
        print(f"{r.method:<32}{r.elo:>8.0f}   [{r.lo:>6.0f}, {r.hi:>6.0f}]{r.battles:>9}"
              f"{r.tasks:>7}")


def _print_sign(t: SignTest) -> None:
    """Print a :class:`SignTest`."""
    print(f"\nsign test, {t.candidate} against {t.reference}, per {t.unit}: {t.wins} wins, "
          f"{t.losses} losses, {t.ties} ties, two-sided p = {t.p_value:.4f}")


def _fintfm_methods(frame: pd.DataFrame) -> list[str]:
    """Methods in a results frame that are not baselines."""
    return sorted(set(frame["method"]) - set(BASELINES))


def compare_runs(a: Path, b: Path, anchor: str = ANCHOR, n_boot: int = 1000, seed: int = 0,
                 unit: str = "task") -> tuple[pd.DataFrame, SignTest]:
    """Elo over both runs' methods, and the sign test of run A's checkpoint against run B's.

    Args:
        a: CSV of the checkpoint under test.
        b: CSV of the reference checkpoint.
        anchor: Elo anchor.
        n_boot: Bootstrap resamples.
        seed: Bootstrap seed.
        unit: Sign-test unit.

    Returns:
        ``(elo_table, sign_test)``. When both runs name their checkpoint alike, each name gets
        its file's stem in brackets.

    Raises:
        ValueError: If a run holds other than exactly one FinTFM method, or the two runs'
            baselines disagree on a shared split (they were not run on identical splits).
    """
    fa, fb = pd.read_csv(a), pd.read_csv(b)
    ma, mb = _fintfm_methods(fa), _fintfm_methods(fb)
    if len(ma) != 1 or len(mb) != 1:
        raise ValueError(f"each run must hold one FinTFM method; {a}: {ma}, {b}: {mb}")
    cand, ref = ma[0], mb[0]
    if cand == ref:
        cand, ref = f"{cand} [{a.stem}]", f"{ref} [{b.stem}]"
        fa.loc[fa["method"] == ma[0], "method"] = cand
        fb.loc[fb["method"] == mb[0], "method"] = ref
    key = ["task_id", "split", "method"]
    base_a = fa[fa["method"].isin(BASELINES)]
    base_b = fb[fb["method"].isin(BASELINES)]
    shared = base_a.merge(base_b, on=key, suffixes=("_a", "_b"))
    bad = shared[~np.isclose(shared["error_a"], shared["error_b"], equal_nan=True)
                 | (shared["n_train_a"] != shared["n_train_b"])]
    if len(bad):
        raise ValueError(f"the runs' baselines differ on {len(bad)} shared split(s), so they "
                         "were not scored on identical splits")
    extra_b = base_b.merge(base_a[key], on=key, how="left", indicator=True)
    extra_b = extra_b.loc[extra_b["_merge"] == "left_only", base_b.columns]
    combined = pd.concat([fa, fb[fb["method"] == ref], extra_b], ignore_index=True)
    return (elo_table(combined, anchor, n_boot, seed),
            sign_test(combined, cand, ref, unit))


def _row_hash(cache_dir: Path, reference_dir: Path, tasks: Sequence[Candidate]) -> int:
    """``--row-hash``: compare every suite dataset's contents with the cached TabArena ones.

    Runs a positive control first: OpenML's original credit-g (dataset 31), whose re-upload is
    TabArena's 46918, must be flagged, or the check is not working and nothing it reports
    counts.

    Returns:
        0 when every task is clean (or the check was skipped, which is printed), 1 when a task
        overlaps or the control is missed.
    """
    references, notes = load_reference_fingerprints(reference_dir)
    for note in notes:
        print(note)
    if not references:
        print("SKIPPED: no TabArena-v0.1 dataset available locally; nothing was compared")
        return 0

    def loader(dataset_id: int, task_id: int | None) -> Callable[[], pd.DataFrame | None]:
        """Features of one dataset, target dropped, or None when it cannot be fetched."""
        def load() -> pd.DataFrame | None:
            """Fetch (cached) and return the features."""
            try:
                target = task_definition(task_id, cache_dir)["target"] if task_id else None
                return openml_frame(dataset_id, target, cache_dir / "openml")[0]
            except Exception as exc:  # noqa: BLE001 -- one dataset's failure is one line
                print(f"  data {dataset_id}: fetch failed ({type(exc).__name__}: {exc})")
                return None
        return load

    print("\npositive control (must read OVERLAP):")
    control = row_hash_report([("credit-g (OpenML 31)", loader(31, None))], references)
    if not control:
        print("CONTROL MISSED: the original of a TabArena dataset was not flagged; the check "
              "below cannot be trusted")
        return 1
    print(f"\nsuite ({len(tasks)} tasks):")
    hits = row_hash_report([(c.name, loader(c.dataset_id, c.task_id)) for c in tasks], references)
    print(f"\n{len(hits)} of {len(tasks)} suite datasets overlap a cached TabArena-v0.1 dataset"
          + (f": {', '.join(hits)}" if hits else ""))
    return 1 if hits else 0


def _select_tasks(ids: str | None) -> list[Candidate]:
    """Suite entries by comma-separated task id; the whole suite when None.

    Raises:
        SystemExit: For an id outside the suite, so an excluded task is never scored under
            the proxy's name.
    """
    if not ids:
        return list(SUITE)
    by_id = {c.task_id: c for c in SUITE}
    wanted = [int(t) for t in ids.split(",") if t.strip()]
    outside = [t for t in wanted if t not in by_id]
    if outside:
        raise SystemExit(f"task(s) {outside} are not in the proxy suite (see --dry-run)")
    return [by_id[t] for t in wanted]


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; see the module docstring for the options' purpose."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="print the suite, the exclusions and their reasons")
    parser.add_argument("--task-ids", action="store_true",
                        help="print the suite's OpenML task ids, comma-separated, for the runner")
    parser.add_argument("--verify-licences", action="store_true",
                        help="re-fetch every candidate's OpenML licence field (network)")
    parser.add_argument("--row-hash", action="store_true",
                        help="fetch every suite dataset and compare its rows and columns with "
                             "the TabArena-v0.1 datasets in the local OpenML cache (network)")
    parser.add_argument("--tabarena-cache", type=Path, default=TABARENA_OPENML_CACHE,
                        help="openml-python dataset cache holding TabArena's datasets (read only)")
    parser.add_argument("--cache-dir", type=Path, default=PROXY_CACHE,
                        help="runner cache: OpenML downloads and baseline scores")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="score a checkpoint and the baselines on the suite")
    run.add_argument("--model", default=None, help="FinTFM checkpoint; omit to cache baselines")
    run.add_argument("--out", type=Path, required=True, help="per-task-per-split CSV")
    run.add_argument("--config", default=None, help="YAML whose inference section builds FinTFM")
    run.add_argument("--name", default=None, help="FinTFM's method name in the CSV")
    run.add_argument("--tasks", default=None, help="comma-separated suite task ids (default all)")
    run.add_argument("--n-splits", type=int, default=N_SPLITS,
                     help="folds of OpenML's repeat 0 to score per task")
    run.add_argument("--no-baselines", action="store_true", help="FinTFM rows only")
    run.add_argument("--n-jobs", type=int, default=1, help="threads for trees and LightGBM")
    run.add_argument("--device", default="cpu")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--n-boot", type=int, default=1000, help="Elo bootstrap resamples")
    run.add_argument(REPORTING_FLAG, action="store_true",
                     help="permit a selection-forbidden dataset, for reporting only")

    cmp_ = sub.add_parser("compare", help="Elo and sign test: checkpoint A against reference B")
    cmp_.add_argument("a", type=Path, help="run CSV of the checkpoint under test")
    cmp_.add_argument("b", type=Path, help="run CSV of the reference checkpoint")
    cmp_.add_argument("--anchor", default=ANCHOR)
    cmp_.add_argument("--n-boot", type=int, default=1000)
    cmp_.add_argument("--seed", type=int, default=0)
    cmp_.add_argument("--unit", choices=("task", "split"), default="task")
    args = parser.parse_args(argv)

    if args.command == "run":
        from fintfm.config import load_config

        frame = run_suite(
            args.model, args.out, tasks=_select_tasks(args.tasks), cfg=load_config(args.config),
            name=args.name, n_splits=args.n_splits, baselines=not args.no_baselines,
            cache_dir=args.cache_dir, n_jobs=args.n_jobs, device=args.device, seed=args.seed,
            allow_reporting=args.allow_tabarena_for_reporting,
        )
        n_ok = int((frame["status"] == "ok").sum())
        print(f"\n{n_ok} of {len(frame)} rows scored; wrote {args.out}")
        if (frame["method"] == ANCHOR).any() and n_ok:
            _print_elo(elo_table(frame, n_boot=args.n_boot, seed=args.seed), ANCHOR, args.n_boot)
        return 0
    if args.command == "compare":
        table, test = compare_runs(args.a, args.b, args.anchor, args.n_boot, args.seed, args.unit)
        _print_elo(table, args.anchor, args.n_boot)
        _print_sign(test)
        return 0
    if args.task_ids:
        print(",".join(str(c.task_id) for c in SUITE))
    if args.verify_licences:
        return 1 if _verify_licences(CANDIDATES) else 0
    if args.row_hash:
        return _row_hash(args.cache_dir, args.tabarena_cache, SUITE)
    if args.dry_run or not args.task_ids:
        _dry_run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
