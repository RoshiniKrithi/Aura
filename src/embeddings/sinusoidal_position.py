"""Sinusoidal Positional Encoding Implementation for Aura LLM Architecture.

Implements Vaswani et al. (2017) fixed sine and cosine frequency positional encodings
registered as a PyTorch non-trainable buffer.
"""

import math
import logging
from typing import Optional
import torch
import torch.nn as nn

from src.embeddings.exceptions import EmbeddingValidationError
from src.embeddings.position_config import PositionEmbeddingConfig
from src.embeddings.position_validator import PositionEmbeddingValidator

logger = logging.getLogger(__name__)


class SinusoidalPositionEmbedding(nn.Module):
    """Production-grade Sinusoidal Positional Encoding module (Vaswani et al., 2017).

    Design Decisions:
        - Calculates fixed sine/cosine frequency waves per dimension pair.
        - Registered as a PyTorch buffer (`register_buffer`) so weights automatically move with
          module device (.to(device)) without requiring gradients or optimizer updates.

    Time Complexity:
        O(B * T * d) slicing operations during forward pass.

    Space Complexity:
        O(M * d) non-trainable buffer memory.
    """

    def __init__(
        self,
        config: Optional[PositionEmbeddingConfig] = None,
        max_sequence_length: Optional[int] = None,
        d_model: Optional[int] = None,
    ) -> None:
        """Initializes SinusoidalPositionEmbedding module.

        Args:
            config: Optional PositionEmbeddingConfig dataclass.
            max_sequence_length: Maximum sequence capacity M.
            d_model: Dense embedding dimension d.
        """
        super().__init__()

        cfg = config or PositionEmbeddingConfig()

        self.max_sequence_length = (
            max_sequence_length if max_sequence_length is not None else cfg.max_sequence_length
        )
        self.d_model = d_model if d_model is not None else cfg.d_model

        # 1. Construct Sinusoidal Matrix PE of shape (max_sequence_length, d_model)
        pe = torch.zeros(self.max_sequence_length, self.d_model, dtype=torch.float32)
        position = torch.arange(0, self.max_sequence_length, dtype=torch.float32).unsqueeze(1)

        # Divisor term: 10000 ^ (2i / d_model) = exp(2i * -log(10000) / d_model)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register non-trainable PyTorch buffer
        self.register_buffer("pe", pe, persistent=True)

        self.validator = PositionEmbeddingValidator(
            max_sequence_length=self.max_sequence_length, d_model=self.d_model
        )

        logger.info(
            "Instantiated SinusoidalPositionEmbedding: MaxSeqLen=%d, d_model=%d",
            self.max_sequence_length,
            self.d_model,
        )

    def forward(
        self,
        position_ids: Optional[torch.Tensor] = None,
        sequence_length: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Forward pass retrieving fixed sinusoidal position vectors.

        Args:
            position_ids: Optional LongTensor of shape (B, T) or (T,).
            sequence_length: Optional sequence length integer T.
            batch_size: Optional batch size B to expand 1D vectors to (B, T, d).

        Returns:
            Sinusoidal positional encoding FloatTensor of shape (B, T, d_model) or (T, d_model).
        """
        if position_ids is None:
            if sequence_length is None:
                raise EmbeddingValidationError(
                    "Either position_ids tensor or sequence_length integer must be provided."
                )

            val_res = self.validator.validate_sequence_length(sequence_length)
            if not val_res.is_valid:
                raise EmbeddingValidationError(
                    f"Sinusoidal sequence length validation failed: {val_res.errors}"
                )

            embeddings = self.pe[:sequence_length]  # Shape: (T, d_model)

            if batch_size is not None and batch_size > 0:
                embeddings = embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # (B, T, d_model)
            return embeddings
        else:
            val_res = self.validator.validate_position_ids(position_ids)
            if not val_res.is_valid:
                raise EmbeddingValidationError(
                    f"Sinusoidal position IDs validation failed: {val_res.errors}"
                )

            # Indexed slicing using buffer
            return self.pe[position_ids]

    def extra_repr(self) -> str:
        """Provides readable PyTorch representation."""
        return f"max_sequence_length={self.max_sequence_length}, d_model={self.d_model}"
