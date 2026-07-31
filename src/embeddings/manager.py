"""Embedding Parameter Lifecycle Manager for Aura LLM Pipeline.

Provides functions for weight freezing/unfreezing, disk saving/loading,
and weight tying (sharing embedding weights with the Language Model head).
"""

import logging
from pathlib import Path
from typing import Union
import torch
import torch.nn as nn

from src.embeddings.embedding_layer import EmbeddingLayer
from src.embeddings.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages state, disk serialization, parameter gradient status, and weight tying for EmbeddingLayer.

    Design Decisions:
        - Decouples parameter manipulation from core PyTorch forward pass logic.
        - Provides explicit weight tying logic: setting output linear projection weight matrix
          to reference the input embedding weight matrix (Press & Wolf, 2017).
    """

    def __init__(self, layer: EmbeddingLayer) -> None:
        """Initializes manager with target EmbeddingLayer.

        Args:
            layer: Target EmbeddingLayer module instance.
        """
        self.layer = layer

    def freeze(self) -> None:
        """Freezes embedding weight parameter updates (disables gradients)."""
        self.layer.weight.requires_grad = False
        logger.info("Froze embedding weights (requires_grad = False).")

    def unfreeze(self) -> None:
        """Unfreezes embedding weight parameter updates (enables gradients)."""
        self.layer.weight.requires_grad = True
        logger.info("Unfroze embedding weights (requires_grad = True).")

    def tie_weights(self, output_linear: nn.Linear) -> None:
        """Ties (shares) embedding weight tensor with an output linear projection head.

        Used in GPT models to reduce model parameters by sharing embedding matrix E
        with the final vocabulary logits projection head: Linear(d_model -> vocab_size).

        Args:
            output_linear: PyTorch nn.Linear output head of shape (vocab_size, d_model).

        Raises:
            EmbeddingError: If dimensions do not match layer weight shape.
        """
        if output_linear.weight.shape != self.layer.weight.shape:
            raise EmbeddingError(
                f"Weight tying dimension mismatch! Embedding weight shape: {tuple(self.layer.weight.shape)}, "
                f"Linear weight shape: {tuple(output_linear.weight.shape)}"
            )

        output_linear.weight = self.layer.weight
        logger.info("Successfully tied embedding weight matrix with LM output projection layer.")

    def save_weights(self, file_path: Union[Path, str]) -> Path:
        """Saves embedding weight tensor to disk.

        Args:
            file_path: Output file path.

        Returns:
            Resolved Path reference.
        """
        path = Path(file_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "weight": self.layer.weight.detach().cpu(),
            "vocab_size": self.layer.vocab_size,
            "d_model": self.layer.d_model,
            "initializer": self.layer.initializer,
            "scale_by_sqrt_d_model": self.layer.scale_by_sqrt_d_model,
            "pad_idx": self.layer.pad_idx,
        }

        try:
            torch.save(state, path)
            logger.info("Saved embedding weights state to: %s", path)
            return path
        except Exception as e:
            raise EmbeddingError(f"Failed to save embedding weights to {path}: {str(e)}") from e

    def load_weights(self, file_path: Union[Path, str], strict: bool = True) -> None:
        """Loads embedding weights from disk into current EmbeddingLayer.

        Args:
            file_path: Input file path.
            strict: If True, enforces strict dimension equality.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise EmbeddingError(f"Embedding weights file not found: {path}")

        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
            saved_weight = state["weight"]

            if saved_weight.shape != self.layer.weight.shape:
                if strict:
                    raise EmbeddingError(
                        f"Saved weight shape {tuple(saved_weight.shape)} does not match current layer shape {tuple(self.layer.weight.shape)}"
                    )

            with torch.no_grad():
                self.layer.weight.copy_(saved_weight)

            logger.info("Successfully loaded embedding weights from: %s", path)
        except Exception as e:
            raise EmbeddingError(f"Failed to load embedding weights from {path}: {str(e)}") from e
