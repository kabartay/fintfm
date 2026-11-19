"""A linear readout on a fintfm network's pre-head representation.

:class:`LinearReadoutModel` fits a fintfm in-context network on a table and decides with a linear
probe (a logistic regression or a ridge) on the network's pre-head representation instead of its
own head. :func:`entry_defaults` gives the measured entry's settings per problem type.
"""

from fintfm.readout.checkpoints import (
    BINARY_CHECKPOINT,
    MULTICLASS_CHECKPOINT,
    REGRESSION_CHECKPOINT,
    resolve_checkpoint,
)
from fintfm.readout.entry import entry_defaults
from fintfm.readout.estimator import LinearReadoutModel

__all__ = [
    "BINARY_CHECKPOINT",
    "MULTICLASS_CHECKPOINT",
    "REGRESSION_CHECKPOINT",
    "LinearReadoutModel",
    "entry_defaults",
    "resolve_checkpoint",
]
