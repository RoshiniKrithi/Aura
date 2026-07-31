"""Aura Self-Attention and Multi-Head Attention Module.

WHY THIS FILE EXISTS:
    Root entry point for Single-Head and Multi-Head Causal Self-Attention operations in Aura architecture.
    Exports MultiHeadAttention, MultiHeadAttentionConfig, AttentionHead, AttentionOutputProjection,
    SelfAttention, AttentionConfig, AttentionMask, AttentionValidator, AttentionStatistics,
    AttentionUtilities, and AttentionFactory.

WHY THIS IMPLEMENTATION WAS CHOSEN:
    Centralized exports enable downstream Transformer Blocks (Phase 11) to import multi-head attention primitives
    directly (`from src.attention import MultiHeadAttention, MultiHeadAttentionConfig`).
"""

from src.attention.attention_head import AttentionHead
from src.attention.config import AttentionConfig
from src.attention.exceptions import (
    AttentionConfigError,
    AttentionError,
    AttentionValidationError,
)
from src.attention.factory import AttentionFactory
from src.attention.mask import AttentionMask
from src.attention.multi_head import MultiHeadAttention
from src.attention.multi_head_config import MultiHeadAttentionConfig
from src.attention.output_projection import AttentionOutputProjection
from src.attention.single_head import SelfAttention
from src.attention.statistics import AttentionStatistics, AttentionStats
from src.attention.utilities import AttentionUtilities
from src.attention.validator import AttentionValidationResult, AttentionValidator

__all__ = [
    "AttentionConfig",
    "MultiHeadAttentionConfig",
    "SelfAttention",
    "MultiHeadAttention",
    "AttentionHead",
    "AttentionOutputProjection",
    "AttentionMask",
    "AttentionValidator",
    "AttentionValidationResult",
    "AttentionStatistics",
    "AttentionStats",
    "AttentionUtilities",
    "AttentionFactory",
    "AttentionError",
    "AttentionValidationError",
    "AttentionConfigError",
]
