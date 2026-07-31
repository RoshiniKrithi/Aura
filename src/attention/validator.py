"""Attention Input and Value Validator for Aura LLM Architecture.

Enforces 3D shape constraints (B, T, d), embedding dimension matching, non-empty batch checks,
and numerical instability detection (NaN / Inf).
"""

from dataclasses import dataclass, field
import logging
from typing import List, Optional
import torch

from src.attention.exceptions import AttentionValidationError

logger = logging.getLogger(__name__)


@dataclass
class AttentionValidationResult:
    """Summary diagnostic report output from attention validation checks."""

    is_valid: bool
    batch_size: int = 0
    sequence_length: int = 0
    d_model: int = 0
    has_nan: bool = False
    has_inf: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class AttentionValidator:
    """Validates input tensors and attention score matrices before and during attention computation.

    Design Decisions:
        - Prevents silent training failures by checking for NaN and Inf values.
        - Verifies 3D tensor shape integrity (B, T, d_model).

    Time Complexity:
        O(1) shape checks; O(B * T * d) torch.isnan / torch.isinf reduction scans.

    Space Complexity:
        O(1) scalar memory overhead.
    """

    def __init__(self, d_model: int) -> None:
        """Initializes validator with target model dimension.

        Args:
            d_model: Expected input/output feature dimension d_model.
        """
        if d_model <= 0:
            raise AttentionValidationError(f"d_model must be positive, got {d_model}")
        self.d_model = d_model

    @staticmethod
    def validate_multi_head_config(d_model: int, n_heads: int) -> None:
        """Validates that d_model is evenly divisible by n_heads.

        Args:
            d_model: Feature dimension d_model.
            n_heads: Number of attention heads H.

        Raises:
            AttentionValidationError: If d_model % n_heads != 0 or parameters are <= 0.
        """
        if d_model <= 0:
            raise AttentionValidationError(f"d_model must be positive, got {d_model}")
        if n_heads <= 0:
            raise AttentionValidationError(f"n_heads must be positive, got {n_heads}")
        if d_model % n_heads != 0:
            raise AttentionValidationError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})."
            )

    def validate_input_embeddings(
        self, x: torch.Tensor, check_values: bool = True
    ) -> AttentionValidationResult:
        """Validates input sequence representation tensor X.

        Args:
            x: Input FloatTensor of shape (B, T, d_model) or (T, d_model).
            check_values: If True, scans tensor for NaN or Inf entries.

        Returns:
            AttentionValidationResult summary container.
        """
        result = AttentionValidationResult(is_valid=True)

        if not isinstance(x, torch.Tensor):
            result.is_valid = False
            result.errors.append(f"Input must be a PyTorch Tensor, got {type(x).__name__}")
            return result

        if x.ndim not in (2, 3):
            result.is_valid = False
            result.errors.append(
                f"Input tensor must be 2D (T, d) or 3D (B, T, d), got shape {tuple(x.shape)}"
            )
            return result

        if x.numel() == 0:
            result.is_valid = False
            result.errors.append("Empty input tensor received.")
            return result

        if x.ndim == 3:
            result.batch_size, result.sequence_length, feat_dim = x.shape
        else:
            result.batch_size = 1
            result.sequence_length, feat_dim = x.shape

        result.d_model = feat_dim

        # Dimension matching check
        if feat_dim != self.d_model:
            result.is_valid = False
            result.errors.append(
                f"Feature dimension mismatch! Input feature dim ({feat_dim}) does not match configured d_model ({self.d_model})."
            )

        # Numerical stability check (NaN / Inf)
        if check_values:
            if torch.isnan(x).any().item():
                result.is_valid = False
                result.has_nan = True
                result.errors.append("Input tensor contains NaN values!")

            if torch.isinf(x).any().item():
                result.is_valid = False
                result.has_inf = True
                result.errors.append("Input tensor contains Inf / -Inf values!")

        if not result.is_valid:
            logger.error("Attention input validation failed: %s", result.errors)

        return result
