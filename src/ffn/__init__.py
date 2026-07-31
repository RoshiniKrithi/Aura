"""Aura Transformer Feed Forward Network (FFN / MLP) Module.

WHY THIS FILE EXISTS:
    Root entry point for point-wise Feed Forward Network (MLP) operations in Aura architecture.
    Exports FeedForwardNetwork, FeedForwardConfig, FeedForwardInitializer, FeedForwardValidator,
    FeedForwardStatistics, FeedForwardUtilities, and FeedForwardFactory.

WHY THIS IMPLEMENTATION WAS CHOSEN:
    Clean Architecture modular exports enable downstream Transformer Blocks (Phase 11) to import
    FFN primitives directly (`from src.ffn import FeedForwardNetwork, FeedForwardConfig`).
"""

from src.ffn.config import FeedForwardConfig
from src.ffn.exceptions import (
    FeedForwardConfigError,
    FeedForwardError,
    FeedForwardValidationError,
)
from src.ffn.factory import FeedForwardFactory
from src.ffn.initializer import FeedForwardInitializer
from src.ffn.network import FeedForwardNetwork
from src.ffn.statistics import FeedForwardStatistics, FeedForwardStats
from src.ffn.utilities import FeedForwardUtilities
from src.ffn.validator import FeedForwardValidationResult, FeedForwardValidator

__all__ = [
    "FeedForwardConfig",
    "FeedForwardNetwork",
    "FeedForwardInitializer",
    "FeedForwardValidator",
    "FeedForwardValidationResult",
    "FeedForwardStatistics",
    "FeedForwardStats",
    "FeedForwardUtilities",
    "FeedForwardFactory",
    "FeedForwardError",
    "FeedForwardValidationError",
    "FeedForwardConfigError",
]
