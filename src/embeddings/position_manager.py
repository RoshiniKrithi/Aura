"""Position Embedding Lifecycle Manager for Aura LLM Architecture.

Provides functions for freezing/unfreezing positional weights, saving/loading to disk,
and combining token embeddings with positional embeddings.
"""

import logging
from pathlib import Path
from typing import Union
import torch
import torch.nn as nn

from src.embeddings.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


class PositionEmbeddingManager:
    """Manages parameter state, disk serialization, and token + positional vector addition.

    Design Decisions:
        - Decouples lifecycle operations from core PyTorch forward pass.
        - Provides broadcast-safe element-wise addition: H_final = H_tok + H_pos.
    """

    def __init__(self, position_module: nn.Module) -> None:
        """Initializes manager.

        Args:
            position_module: Position embedding module instance (Learnable or Sinusoidal).
        """
        self.position_module = position_module

    def freeze(self) -> None:
        """Freezes positional embedding weight parameters."""
        for param in self.position_module.parameters():
            param.requires_grad = False
        logger.info("Froze positional embedding parameters.")

    def unfreeze(self) -> None:
        """Unfreezes positional embedding weight parameters."""
        for param in self.position_module.parameters():
            param.requires_grad = True
        logger.info("Unfroze positional embedding parameters.")

    @staticmethod
    def combine_embeddings(
        token_embeddings: torch.Tensor,
        position_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Performs element-wise addition of token and positional embeddings: H = H_tok + H_pos.

        Args:
            token_embeddings: FloatTensor of shape (B, T, d) or (T, d).
            position_embeddings: FloatTensor of shape (B, T, d) or (T, d).

        Returns:
            Combined FloatTensor of shape matching input token_embeddings.

        Raises:
            EmbeddingError: If tensor dimensions or sequence lengths mismatch.
        """
        if token_embeddings.shape[-1] != position_embeddings.shape[-1]:
            raise EmbeddingError(
                f"Embedding dimension mismatch! Token embedding d_model: {token_embeddings.shape[-1]}, "
                f"Position embedding d_model: {position_embeddings.shape[-1]}"
            )

        if token_embeddings.ndim == 3 and position_embeddings.ndim == 2:
            position_embeddings = position_embeddings.unsqueeze(0)

        try:
            return token_embeddings + position_embeddings
        except Exception as e:
            raise EmbeddingError(
                f"Failed to combine token and positional embeddings: {str(e)}"
            ) from e

    def save_weights(self, file_path: Union[Path, str]) -> Path:
        """Saves positional embedding parameters or buffer to disk."""
        path = Path(file_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        state = self.position_module.state_dict()
        try:
            torch.save(state, path)
            logger.info("Saved positional embedding weights to: %s", path)
            return path
        except Exception as e:
            raise EmbeddingError(
                f"Failed saving position embedding weights to {path}: {str(e)}"
            ) from e

    def load_weights(self, file_path: Union[Path, str]) -> None:
        """Loads positional embedding parameters from disk."""
        path = Path(file_path).resolve()
        if not path.exists():
            raise EmbeddingError(f"Position weights file not found: {path}")

        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
            self.position_module.load_state_dict(state)
            logger.info("Successfully loaded positional embedding weights from: %s", path)
        except Exception as e:
            raise EmbeddingError(
                f"Failed loading position embedding weights from {path}: {str(e)}"
            ) from e
