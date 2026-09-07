"""Synthetic task priors. Every prior returns a :class:`Task`."""

from fintfm.prior.base import Task, TaskBatch, collate
from fintfm.prior.financial import sample_financial_task
from fintfm.prior.mixture import PriorConfig, sample_task
from fintfm.prior.scm import sample_scm_task

__all__ = [
    "PriorConfig",
    "Task",
    "TaskBatch",
    "collate",
    "sample_financial_task",
    "sample_scm_task",
    "sample_task",
]
