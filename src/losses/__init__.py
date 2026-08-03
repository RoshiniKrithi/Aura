"""Loss Subsystem Module for Aura LLM Architecture.

Provides CrossEntropyLossConfig, CrossEntropyLoss, LossFactory,
LossValidator, and LossStatistics.
"""

from src.losses.config import CrossEntropyLossConfig
from src.losses.cross_entropy import CrossEntropyLoss
from src.losses.exceptions import LossConfigError, LossError, LossValidationError
from src.losses.factory import LossFactory
from src.losses.statistics import LossStats, LossStatistics
from src.losses.validator import LossValidationResult, LossValidator

__all__ = [
    "CrossEntropyLossConfig",
    "CrossEntropyLoss",
    "LossFactory",
    "LossValidator",
    "LossValidationResult",
    "LossStatistics",
    "LossStats",
    "LossError",
    "LossValidationError",
    "LossConfigError",
]
