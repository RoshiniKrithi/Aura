"""Positional Embedding Validator for Aura LLM Architecture.

Enforces sequence length boundary checks, negative position index prevention,
and dimension shape integrity.
"""

from dataclasses import dataclass, field
import logging
from typing import List
import torch

from src.embeddings.exceptions import EmbeddingValidationError

logger = logging.getLogger(__name__)


@dataclass
class PositionValidationResult:
    """Diagnostic report output from positional embedding validation checks."""

    is_valid: bool
    sequence_length: int = 0
    max_sequence_length: int = 0
    d_model: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class PositionEmbeddingValidator:
    """Validates sequence lengths and position IDs before performing embedding lookup operations.

    Design Decisions:
        - Strict sequence length bound enforcement (T <= max_sequence_length) to prevent array overflow.
    """

    def __init__(self, max_sequence_length: int, d_model: int) -> None:
        """Initializes validator with capacity constraints.

        Args:
            max_sequence_length: Maximum sequence length capacity M.
            d_model: Dense vector dimension d.
        """
        if max_sequence_length <= 0:
            raise EmbeddingValidationError(
                f"max_sequence_length must be positive, got {max_sequence_length}"
            )
        if d_model <= 0:
            raise EmbeddingValidationError(f"d_model must be positive, got {d_model}")

        self.max_sequence_length = max_sequence_length
        self.d_model = d_model

    def validate_sequence_length(self, seq_len: int) -> PositionValidationResult:
        """Validates scalar sequence length T.

        Args:
            seq_len: Sequence length integer T.

        Returns:
            PositionValidationResult summary container.
        """
        result = PositionValidationResult(
            is_valid=True,
            sequence_length=seq_len,
            max_sequence_length=self.max_sequence_length,
            d_model=self.d_model,
        )

        if seq_len <= 0:
            result.is_valid = False
            result.errors.append(f"Sequence length must be positive (> 0), got {seq_len}")
            return result

        if seq_len > self.max_sequence_length:
            result.is_valid = False
            result.errors.append(
                f"Sequence length ({seq_len}) exceeds maximum sequence capacity ({self.max_sequence_length})."
            )

        return result

    def validate_position_ids(self, position_ids: torch.Tensor) -> PositionValidationResult:
        """Validates position ID tensor against shape, dtype, and bounds.

        Args:
            position_ids: LongTensor containing position indices.

        Returns:
            PositionValidationResult summary container.
        """
        result = PositionValidationResult(
            is_valid=True,
            max_sequence_length=self.max_sequence_length,
            d_model=self.d_model,
        )

        if not isinstance(position_ids, torch.Tensor):
            result.is_valid = False
            result.errors.append(
                f"position_ids must be a PyTorch Tensor, got {type(position_ids).__name__}"
            )
            return result

        if torch.is_floating_point(position_ids):
            result.is_valid = False
            result.errors.append(
                f"position_ids must be integer tensor, got floating-point {position_ids.dtype}"
            )
            return result

        if position_ids.numel() == 0:
            result.is_valid = False
            result.errors.append("Empty position_ids tensor received.")
            return result

        min_val = position_ids.min().item()
        max_val = position_ids.max().item()

        if min_val < 0:
            result.is_valid = False
            result.errors.append(f"Detected negative position index ({min_val}).")

        if max_val >= self.max_sequence_length:
            result.is_valid = False
            result.errors.append(
                f"Detected position index ({max_val}) exceeding max sequence capacity ({self.max_sequence_length - 1})."
            )

        return result
