"""Aura Token and Positional Embedding Module.

WHY THIS FILE EXISTS:
    Root entry point for discrete token ID and sequence position mappings into continuous
    vector space. Exports Token Embedding primitives, Positional Embedding modules
    (Learnable & Sinusoidal), position generators, managers, factories, and the composite
    InputEmbeddingPipeline.
"""

from src.embeddings.config import EmbeddingConfig
from src.embeddings.embedding_layer import EmbeddingLayer
from src.embeddings.exceptions import (
    EmbeddingConfigError,
    EmbeddingError,
    EmbeddingInitializationError,
    EmbeddingValidationError,
)
from src.embeddings.factory import EmbeddingFactory
from src.embeddings.initializer import EmbeddingInitializer
from src.embeddings.input_embedding_pipeline import InputEmbeddingPipeline
from src.embeddings.learnable_position import LearnablePositionEmbedding
from src.embeddings.manager import EmbeddingManager
from src.embeddings.position_config import PositionEmbeddingConfig
from src.embeddings.position_factory import PositionEmbeddingFactory
from src.embeddings.position_initializer import PositionEmbeddingInitializer
from src.embeddings.position_manager import PositionEmbeddingManager
from src.embeddings.position_utilities import PositionEmbeddingUtilities
from src.embeddings.position_validator import (
    PositionEmbeddingValidator,
    PositionValidationResult,
)
from src.embeddings.sinusoidal_position import SinusoidalPositionEmbedding
from src.embeddings.statistics import EmbeddingStatistics, EmbeddingStats
from src.embeddings.utilities import EmbeddingUtilities
from src.embeddings.validator import EmbeddingValidationResult, EmbeddingValidator

__all__ = [
    "EmbeddingConfig",
    "EmbeddingLayer",
    "EmbeddingInitializer",
    "EmbeddingValidator",
    "EmbeddingValidationResult",
    "EmbeddingStatistics",
    "EmbeddingStats",
    "EmbeddingManager",
    "EmbeddingUtilities",
    "EmbeddingFactory",
    "PositionEmbeddingConfig",
    "LearnablePositionEmbedding",
    "SinusoidalPositionEmbedding",
    "PositionEmbeddingInitializer",
    "PositionEmbeddingValidator",
    "PositionValidationResult",
    "PositionEmbeddingUtilities",
    "PositionEmbeddingManager",
    "PositionEmbeddingFactory",
    "InputEmbeddingPipeline",
    "EmbeddingError",
    "EmbeddingValidationError",
    "EmbeddingInitializationError",
    "EmbeddingConfigError",
]
