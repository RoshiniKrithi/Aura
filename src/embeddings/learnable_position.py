"""Learnable Positional Embedding Implementation for Aura LLM Architecture.

Implements absolute learnable positional embeddings (GPT-2 standard) wrapping
a PyTorch parameter matrix P of shape (max_sequence_length, d_model).
"""

import logging
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.embeddings.exceptions import EmbeddingValidationError
from src.embeddings.position_config import PositionEmbeddingConfig
from src.embeddings.position_initializer import PositionEmbeddingInitializer
from src.embeddings.position_validator import PositionEmbeddingValidator

logger = logging.getLogger(__name__)


class LearnablePositionEmbedding(nn.Module):
    """Production-grade GPT-2 style Learnable Positional Embedding module.

    Design Decisions:
        - Stores a learnable Parameter matrix P of shape (max_sequence_length, d_model).
        - Automatically generates position IDs [0, 1, ..., T-1] if sequence length T is provided.
        - Supports explicit 2D position ID tensors (B, T) for custom position offsets.

    Time Complexity:
        O(B * T * d) lookup operations.

    Space Complexity:
        O(M * d) parameter memory where M is max_sequence_length and d is d_model.
    """

    def __init__(
        self,
        config: Optional[PositionEmbeddingConfig] = None,
        max_sequence_length: Optional[int] = None,
        d_model: Optional[int] = None,
        initializer: str = "normal",
        init_range: float = 0.02,
        init_std: float = 0.02,
    ) -> None:
        """Initializes LearnablePositionEmbedding module.

        Args:
            config: Optional PositionEmbeddingConfig dataclass.
            max_sequence_length: Maximum sequence length capacity M.
            d_model: Dense embedding vector dimension d.
            initializer: Weight initialization method.
            init_range: Range for uniform initializations.
            init_std: Standard deviation for normal initializations.
        """
        super().__init__()

        cfg = config or PositionEmbeddingConfig()

        self.max_sequence_length = (
            max_sequence_length if max_sequence_length is not None else cfg.max_sequence_length
        )
        self.d_model = d_model if d_model is not None else cfg.d_model
        self.initializer = initializer if config is None else cfg.initializer
        self.init_range = init_range if config is None else cfg.init_range
        self.init_std = init_std if config is None else cfg.init_std

        # 1. Parameter Matrix Creation: (max_sequence_length, d_model)
        self.weight = nn.Parameter(torch.empty(self.max_sequence_length, self.d_model))

        # 2. Weight Initialization
        PositionEmbeddingInitializer.initialize(
            weight=self.weight,
            method=self.initializer,
            init_range=self.init_range,
            init_std=self.init_std,
        )

        # 3. Validator
        self.validator = PositionEmbeddingValidator(
            max_sequence_length=self.max_sequence_length, d_model=self.d_model
        )

        logger.info(
            "Instantiated LearnablePositionEmbedding: MaxSeqLen=%d, d_model=%d, Initializer=%s",
            self.max_sequence_length,
            self.d_model,
            self.initializer,
        )

    def forward(
        self,
        position_ids: Optional[torch.Tensor] = None,
        sequence_length: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Forward pass generating positional embedding vectors.

        Args:
            position_ids: Optional LongTensor of shape (B, T) or (T,).
            sequence_length: Optional sequence length T (used if position_ids is None).
            batch_size: Optional batch size B to expand 1D position vectors to 2D (B, T, d).

        Returns:
            Positional embedding FloatTensor of shape (B, T, d_model) or (T, d_model).

        Raises:
            EmbeddingValidationError: If position_ids or sequence_length is invalid.
        """
        if position_ids is None:
            if sequence_length is None:
                raise EmbeddingValidationError(
                    "Either position_ids tensor or sequence_length integer must be provided."
                )

            val_res = self.validator.validate_sequence_length(sequence_length)
            if not val_res.is_valid:
                raise EmbeddingValidationError(
                    f"Position sequence length validation failed: {val_res.errors}"
                )

            # Generate default position IDs: [0, 1, ..., T-1]
            pos_ids = torch.arange(0, sequence_length, dtype=torch.long, device=self.weight.device)
            embeddings = F.embedding(pos_ids, self.weight)  # Shape: (T, d_model)

            if batch_size is not None and batch_size > 0:
                embeddings = embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # (B, T, d_model)
            return embeddings

        else:
            val_res = self.validator.validate_position_ids(position_ids)
            if not val_res.is_valid:
                raise EmbeddingValidationError(
                    f"Position IDs validation failed: {val_res.errors}"
                )

            return F.embedding(position_ids, self.weight)

    def extra_repr(self) -> str:
        """Provides readable PyTorch module representation."""
        return (
            f"max_sequence_length={self.max_sequence_length}, d_model={self.d_model}, "
            f"initializer='{self.initializer}'"
        )
