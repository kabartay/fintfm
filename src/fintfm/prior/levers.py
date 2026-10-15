"""Command-line wiring for the audit Phase 5 pretraining levers (AU5.4-AU5.6, AU5.15).

Kept beside the prior rather than inlined into ``fintfm-train`` so the training entry point
and ``scripts/dump_prior_tasks.py`` build the same :class:`~fintfm.prior.mixture.PriorConfig`
from the same flags -- a task dump is only usable if the run consuming it builds an identical
prior, and two copies of the flag parsing would drift. Every flag defaults to ``None`` so the
``prior`` section of the configuration supplies the value and an explicit flag still wins.
"""

from __future__ import annotations

import argparse
from typing import Any

from fintfm.config import PriorConfigDefaults


def add_prior_lever_arguments(p: argparse.ArgumentParser) -> None:
    """Add the lever flags to a parser.

    Args:
        p: The parser to extend.
    """
    p.add_argument(
        "--permute-classes",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="relabel each classification task's classes by a random permutation (audit "
        "AU5.5), so no class index acquires a fixed meaning such as 'class 1 is rare'; "
        "defaults to the config's prior.permute_classes, false",
    )
    p.add_argument(
        "--feature-transforms",
        type=str,
        default=None,
        help="comma-separated inference transforms (none,rank,power,winsor) each task draws "
        "one of, fitted on its context rows only (audit AU5.6); defaults to the config's "
        "prior.feature_transforms, empty, which applies none",
    )
    p.add_argument(
        "--max-query-rows",
        type=int,
        default=None,
        help="keep at most this many query rows per task (audit AU5.4), so a long context "
        "drawn from --n-rows-choices does not bring a long query with it; defaults to the "
        "config's prior.max_query_rows, null, which keeps every row",
    )
    p.add_argument(
        "--task-dump",
        type=str,
        default=None,
        help="read pretraining batches from a dump written by scripts/dump_prior_tasks.py "
        "instead of sampling them (audit AU5.15). The dump's prior and batch size must equal "
        "this run's; a 'sequential' dump written with this run's --seed reproduces the live "
        "batches exactly",
    )


def prior_lever_values(args: argparse.Namespace, prior: PriorConfigDefaults) -> dict[str, Any]:
    """Resolve the lever flags against the configuration's ``prior`` section.

    Args:
        args: Parsed arguments from a parser extended by :func:`add_prior_lever_arguments`.
        prior: The loaded configuration's ``prior`` section.

    Returns:
        Keyword arguments for :class:`~fintfm.prior.mixture.PriorConfig`.
    """
    transforms = (
        prior.feature_transforms
        if args.feature_transforms is None
        else tuple(k.strip() for k in args.feature_transforms.split(",") if k.strip())
    )
    return {
        "permute_classes": (
            prior.permute_classes if args.permute_classes is None else args.permute_classes
        ),
        "feature_transforms": tuple(transforms),
        "max_query_rows": (
            prior.max_query_rows if args.max_query_rows is None else args.max_query_rows
        ),
    }
