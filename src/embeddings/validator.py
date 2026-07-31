"""Token Embedding Input Validator for Aura LLM Pipeline.

Enforces boundary checking, index range validation, tensor shape integrity,
and non-empty batch constraints on embedding inputs.
"""

from dataclasses import dataclass, field
import logging
from typing import List, Optional
import torch

from src.embeddings.exceptions import EmbeddingValidationError

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingValidationResult:
    """Summary diagnostic container resulting from embedding validation checks."""

    is_valid: bool
    batch_size: int = 0
    sequence_length: int = 0
    total_tokens: int = 0
    out_of_bounds_count: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class EmbeddingValidator:
    """Validates token ID tensors before performing embedding lookup operations.

    Design Decisions:
        - Prevents CUDA out-of-bounds illegal memory access crashes by catching invalid token IDs early.
        - Verifies tensor dtypes and dimensional constraints.

    Time Complexity:
        O(1) shape checks; O(B * T) min/max reduction for token ID range validation.

    Space Complexity:
        O(1) scalar check memory.
    """

    def __init__(self, vocab_size: int, d_model: int) -> None:
        """Initializes validator with embedding dimensions.

        Args:
            vocab_size: Total vocabulary size V.
            d_model: Dense vector dimension d.
        """
        if vocab_size <= 0:
            raise EmbeddingValidationError(f"vocab_size must be positive, got {vocab_size}")
        if d_model <= 0:
            raise EmbeddingValidationError(f"d_model must be positive, got {d_model}")

        self.vocab_size = vocab_size
        self.d_model = d_model

    def validate_input_ids(
        self, input_ids: torch.Tensor, pad_idx: Optional[int] = None
    ) -> EmbeddingValidationResult:
        """Validates token ID tensor against shape, dtype, and vocabulary bounds.

        Args:
            input_ids: PyTorch Tensor containing discrete integer token IDs.
            pad_idx: Optional padding token index.

        Returns:
            EmbeddingValidationResult summary container.
        """
        result = EmbeddingValidationResult(is_valid=True)

        if not isinstance(input_ids, torch.Tensor):
            result.is_valid = False
            result.errors.append(
                f"input_ids must be a PyTorch Tensor, got {type(input_ids).__name__}"
            )
            return result

        # 1. Dtype Integrity Check (Integer types required)
        if not torch.is_floating_point(input_ids) and input_ids.dtype not in (
            torch.int64,
            torch.int32,
            torch.int16,
            torch.uint8,
        ):
            result.is_valid = False
            result.errors.append(
                f"input_ids must have integer dtype (e.g. torch.long), got {input_ids.dtype}"
            )
            return result
        elif torch.is_floating_point(input_ids):
            result.is_valid = False
            result.errors.append(
                f"input_ids cannot be floating-point tensor, got {input_ids.dtype}"
            )
            return result

        # 2. Shape Integrity Check (1D or 2D expected)
        if input_ids.ndim not in (1, 2):
            result.is_valid = False
            result.errors.append(
                f"input_ids must be 1D (T,) or 2D (B, T) tensor, got shape {tuple(input_ids.shape)}"
            )
            return result

        if input_ids.numel() == 0:
            result.is_valid = False
            result.errors.append(
                f"Empty token input_ids tensor received with shape {tuple(input_ids.shape)}"
            )
            return result

        if input_ids.ndim == 2:
            result.batch_size, result.sequence_length = input_ids.shape
        else:
            result.batch_size = 1
            result.sequence_length = input_ids.size(0)

        result.total_tokens = input_ids.numel()

        # 3. Min/Max Boundary Checks: 0 <= token_id < vocab_size
        min_val = input_ids.min().item()
        max_val = input_ids.max().item()

        if min_val < 0:
            result.is_valid = False
            result.errors.append(
                f"Detected negative token ID ({min_val}). Token IDs must be >= 0."
            )

        if max_val >= self.vocab_size:
            result.is_valid = False
            result.errors.append(
                f"Detected token ID ({max_val}) exceeding vocabulary bound ({self.vocab_size - 1})."
            )

        if not result.is_valid:
            logger.error("Embedding input validation failed: %s", result.errors)

        return result
