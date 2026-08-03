"""Transformer Module for Aura LLM Architecture.

Assembles Residual Connections (Phase 11) and Transformer Decoder Blocks (Phase 12).
"""

from src.transformer.block_config import TransformerBlockConfig
from src.transformer.block_factory import TransformerBlockFactory
from src.transformer.block_statistics import (
    TransformerBlockStats,
    TransformerBlockStatistics,
)
from src.transformer.block_validator import (
    TransformerBlockValidationResult,
    TransformerBlockValidator,
)
from src.transformer.config import ResidualConfig
from src.transformer.exceptions import (
    ResidualConfigError,
    ResidualError,
    ResidualValidationError,
)
from src.transformer.factory import ResidualFactory
from src.transformer.residual import ResidualConnection
from src.transformer.statistics import ResidualStats, ResidualStatistics
from src.transformer.transformer_block import TransformerBlock
from src.transformer.utilities import ResidualUtilities
from src.transformer.validator import ResidualValidationResult, ResidualValidator

__all__ = [
    "TransformerBlockConfig",
    "TransformerBlock",
    "TransformerBlockFactory",
    "TransformerBlockStatistics",
    "TransformerBlockStats",
    "TransformerBlockValidator",
    "TransformerBlockValidationResult",
    "ResidualConfig",
    "ResidualConnection",
    "ResidualFactory",
    "ResidualStatistics",
    "ResidualStats",
    "ResidualUtilities",
    "ResidualValidator",
    "ResidualValidationResult",
    "ResidualError",
    "ResidualValidationError",
    "ResidualConfigError",
]
