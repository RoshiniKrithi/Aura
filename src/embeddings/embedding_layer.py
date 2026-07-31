"""Token Embedding Layer Implementation for Aura LLM Architecture.

Maps discrete token integer IDs to continuous, dense vector representations
using PyTorch parameter lookup tables.
"""

import math
import logging
from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.embeddings.config import EmbeddingConfig
from src.embeddings.initializer import EmbeddingInitializer
from src.embeddings.validator import EmbeddingValidator
from src.embeddings.exceptions import EmbeddingValidationError

logger = logging.getLogger(__name__)


class EmbeddingLayer(nn.Module):
    """Production-grade PyTorch Token Embedding Layer.

    Design Decisions:
        - Wraps a learnable weight matrix E of shape (vocab_size, d_model) as an nn.Parameter.
        - Direct table lookup using PyTorch C++ optimized functional core (F.embedding).
        - Optional scaling by sqrt(d_model) for Transformer models (Vaswani et al.).
        - Built-in validation to prevent CUDA out-of-bounds illegal memory access crashes.

    Time Complexity:
        O(B * T * d) tensor lookup and scaling operations.

    Space Complexity:
        O(V * d) parameter memory for embedding weight matrix E,
        O(B * T * d) output tensor allocation.
    """

    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        vocab_size: Optional[int] = None,
        d_model: Optional[int] = None,
        initializer: str = "normal",
        init_range: float = 0.02,
        init_std: float = 0.02,
        scale_by_sqrt_d_model: bool = False,
        pad_idx: Optional[int] = None,
    ) -> None:
        """Initializes EmbeddingLayer.

        Args:
            config: Optional EmbeddingConfig dataclass.
            vocab_size: Vocabulary size V (overrides config if provided).
            d_model: Dense embedding dimension d (overrides config if provided).
            initializer: Weight initialization strategy name.
            init_range: Range for uniform initializations.
            init_std: Standard deviation for normal initializations.
            scale_by_sqrt_d_model: If True, scales outputs by sqrt(d_model).
            pad_idx: Padding index to zero out vectors and gradients.
        """
        super().__init__()

        cfg = config or EmbeddingConfig()

        self.vocab_size = vocab_size if vocab_size is not None else cfg.vocab_size
        self.d_model = d_model if d_model is not None else cfg.d_model
        self.initializer = initializer if config is None else cfg.initializer
        self.init_range = init_range if config is None else cfg.init_range
        self.init_std = init_std if config is None else cfg.init_std
        self.scale_by_sqrt_d_model = (
            scale_by_sqrt_d_model if config is None else cfg.scale_by_sqrt_d_model
        )
        self.pad_idx = pad_idx if config is None else cfg.pad_idx

        # 1. Parameter Creation
        self.weight = nn.Parameter(torch.empty(self.vocab_size, self.d_model))

        # 2. Weight Initialization
        EmbeddingInitializer.initialize(
            weight=self.weight,
            method=self.initializer,
            init_range=self.init_range,
            init_std=self.init_std,
            pad_idx=self.pad_idx,
        )

        # 3. Input Validator
        self.validator = EmbeddingValidator(
            vocab_size=self.vocab_size, d_model=self.d_model
        )

        logger.info(
            "Instantiated EmbeddingLayer: VocabSize=%d, d_model=%d, Initializer=%s, PadIdx=%s",
            self.vocab_size,
            self.d_model,
            self.initializer,
            self.pad_idx,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass looking up token IDs in embedding parameter matrix E.

        Args:
            input_ids: LongTensor of token IDs of shape (B, T) or (T,).

        Returns:
            Dense vector representations FloatTensor of shape (B, T, d_model) or (T, d_model).

        Raises:
            EmbeddingValidationError: If input validation checks fail.
        """
        # Validate Input Integrity
        val_result = self.validator.validate_input_ids(
            input_ids, pad_idx=self.pad_idx
        )
        if not val_result.is_valid:
            raise EmbeddingValidationError(
                f"Embedding forward input validation failed: {val_result.errors}"
            )

        # Lookup Embedding Vectors: (B, T) -> (B, T, d_model)
        embeddings = F.embedding(
            input_ids,
            self.weight,
            padding_idx=self.pad_idx,
        )

        # Optional Scaling by sqrt(d_model)
        if self.scale_by_sqrt_d_model:
            embeddings = embeddings * math.sqrt(self.d_model)

        return embeddings

    def extra_repr(self) -> str:
        """Provides readable PyTorch string representation."""
        return (
            f"vocab_size={self.vocab_size}, d_model={self.d_model}, "
            f"initializer='{self.initializer}', scale_by_sqrt_d_model={self.scale_by_sqrt_d_model}"
        )
