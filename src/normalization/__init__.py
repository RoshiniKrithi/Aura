"""Aura Layer Normalization Module.

WHY THIS FILE EXISTS:
    Root entry point for Layer Normalization operations in Aura architecture.
    Exports LayerNormalization, LayerNormConfig, LayerNormValidator, LayerNormStatistics,
    LayerNormUtilities, and LayerNormFactory.

WHY THIS IMPLEMENTATION WAS CHOSEN:
    Clean Architecture modular exports enable downstream Residual Connections (Phase 11) and Transformer Blocks
    to import normalization primitives directly (`from src.normalization import LayerNormalization, LayerNormConfig`).
"""

from src.normalization.config import LayerNormConfig
from src.normalization.exceptions import (
    LayerNormConfigError,
    LayerNormError,
    LayerNormValidationError,
)
from src.normalization.factory import LayerNormFactory
from src.normalization.layer_norm import LayerNormalization
from src.normalization.statistics import LayerNormStatistics, LayerNormStats
from src.normalization.utilities import LayerNormUtilities
from src.normalization.validator import LayerNormValidationResult, LayerNormValidator

__all__ = [
    "LayerNormConfig",
    "LayerNormalization",
    "LayerNormValidator",
    "LayerNormValidationResult",
    "LayerNormStatistics",
    "LayerNormStats",
    "LayerNormUtilities",
    "LayerNormFactory",
    "LayerNormError",
    "LayerNormValidationError",
    "LayerNormConfigError",
]
