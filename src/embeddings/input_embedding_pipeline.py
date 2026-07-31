"""Composite Input Embedding Pipeline for Aura LLM Architecture.

Combines Token Embeddings (Phase 5) + Positional Embeddings (Phase 6) + Dropout
into unified input vector representations ready for Transformer blocks.
"""

import logging
from typing import Any, Optional, Tuple
import torch
import torch.nn as nn

from src.embeddings.embedding_layer import EmbeddingLayer
from src.embeddings.exceptions import EmbeddingValidationError
from src.embeddings.position_factory import PositionEmbeddingFactory
from src.embeddings.position_manager import PositionEmbeddingManager

logger = logging.getLogger(__name__)


class InputEmbeddingPipeline(nn.Module):
    """Composite PyTorch Module delivering final input embeddings for Transformer blocks.

    Design Decisions:
        - Computes H_tok = TokenEmbedding(input_ids) of shape (B, T, d).
        - Computes H_pos = PositionEmbedding(seq_len=T, batch_size=B) of shape (B, T, d).
        - Sums H_combined = H_tok + H_pos and applies dropout.

    Time Complexity:
        O(B * T * d) lookup and element-wise tensor addition ops.

    Space Complexity:
        O((V + M) * d) parameter memory for token and position weight matrices.
    """

    def __init__(
        self,
        token_embedding: EmbeddingLayer,
        position_embedding: nn.Module,
        dropout: float = 0.1,
    ) -> None:
        """Initializes InputEmbeddingPipeline.

        Args:
            token_embedding: Instantiated EmbeddingLayer instance.
            position_embedding: Instantiated positional embedding module (Learnable or Sinusoidal).
            dropout: Dropout probability applied to final combined representation.
        """
        super().__init__()

        self.token_embedding = token_embedding
        self.position_embedding = position_embedding
        self.dropout = nn.Dropout(p=dropout)

        # Validate dimension alignment between token and positional embeddings
        tok_dim = self.token_embedding.d_model
        pos_dim = getattr(self.position_embedding, "d_model", tok_dim)
        if tok_dim != pos_dim:
            raise EmbeddingValidationError(
                f"Dimension mismatch between Token Embedding ({tok_dim}) and Position Embedding ({pos_dim})"
            )

        logger.info(
            "Instantiated InputEmbeddingPipeline: d_model=%d, Dropout=%.2f",
            tok_dim,
            dropout,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass outputting combined token + positional embeddings.

        Args:
            input_ids: LongTensor of shape (B, T) or (T,).
            position_ids: Optional explicit position IDs tensor.

        Returns:
            Combined FloatTensor of shape (B, T, d_model) or (T, d_model).
        """
        # 1. Token Embedding Lookup: (B, T) -> (B, T, d_model)
        token_embeds = self.token_embedding(input_ids)

        # 2. Position Embedding Lookup: (B, T) -> (B, T, d_model)
        if input_ids.ndim == 2:
            b_size, seq_len = input_ids.shape
        else:
            b_size = 1
            seq_len = input_ids.size(0)

        if position_ids is not None:
            pos_embeds = self.position_embedding(position_ids=position_ids)
        else:
            pos_embeds = self.position_embedding(
                sequence_length=seq_len, batch_size=b_size if input_ids.ndim == 2 else None
            )

        # 3. Element-wise Addition: H_combined = H_tok + H_pos
        combined = PositionEmbeddingManager.combine_embeddings(token_embeds, pos_embeds)

        # 4. Dropout
        return self.dropout(combined)

    @classmethod
    def from_config(cls, config: Any) -> "InputEmbeddingPipeline":
        """Factory helper instantiating pipeline directly from AppConfig."""
        from src.embeddings.factory import EmbeddingFactory
        from src.utils.config import AppConfig

        tok_embed = EmbeddingFactory.create_embedding_layer(config)
        pos_embed = PositionEmbeddingFactory.create_position_embedding(config)
        dropout_p = config.position_embedding.dropout

        return cls(
            token_embedding=tok_embed,
            position_embedding=pos_embed,
            dropout=dropout_p,
        )
