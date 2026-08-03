"""LM Head Input Tensor Validator for Aura Architecture.

Validates hidden state tensor dimensions (B, T, d_model), feature matching,
tensor types, and detects NaN / Inf numerical instabilities.
"""

from dataclasses import dataclass, field
import logging
from typing import List
import torch

from src.models.exceptions import LMHeadValidationError

logger = logging.getLogger(__name__)


@dataclass
class LMHeadValidationResult:
    """Summary diagnostic report output from LM Head validation checks."""

    is_valid: bool
    batch_size: int = 0
    sequence_length: int = 0
    d_model: int = 0
    has_nan: bool = False
    has_inf: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class LMHeadValidator:
    """Validates input hidden state tensors for LanguageModelingHead.

    Time Complexity:
        O(1) shape and dimension checks; O(B * T * d) when scanning for NaN/Inf.

    Space Complexity:
        O(1) scalar memory overhead.
    """

    def __init__(self, d_model: int = 768, vocab_size: int = 50257) -> None:
        """Initializes validator with expected hidden dimension and vocabulary size.

        Args:
            d_model: Expected input feature dimension.
            vocab_size: Output vocabulary size.
        """
        if d_model <= 0:
            raise LMHeadValidationError(f"d_model must be positive, got {d_model}")
        if vocab_size <= 0:
            raise LMHeadValidationError(f"vocab_size must be positive, got {vocab_size}")

        self.d_model = d_model
        self.vocab_size = vocab_size

    def validate_hidden_states(
        self, hidden_states: torch.Tensor, check_nan_inf: bool = True
    ) -> LMHeadValidationResult:
        """Validates input hidden_states FloatTensor shape and numerical values.

        Args:
            hidden_states: Input FloatTensor of shape (B, T, d_model) or (T, d_model).
            check_nan_inf: If True, scans tensor for NaN and Inf values.

        Returns:
            LMHeadValidationResult diagnostic summary object.
        """
        result = LMHeadValidationResult(is_valid=True)

        if not isinstance(hidden_states, torch.Tensor):
            result.is_valid = False
            result.errors.append(f"hidden_states must be a PyTorch Tensor, got {type(hidden_states).__name__}")
            return result

        if hidden_states.ndim not in (2, 3):
            result.is_valid = False
            result.errors.append(
                f"hidden_states must be 2D (T, d_model) or 3D (B, T, d_model), got shape {tuple(hidden_states.shape)}"
            )
            return result

        if hidden_states.ndim == 3:
            result.batch_size, result.sequence_length, result.d_model = hidden_states.shape
        else:
            result.batch_size = 1
            result.sequence_length, result.d_model = hidden_states.shape

        if result.d_model != self.d_model:
            result.is_valid = False
            result.errors.append(
                f"Feature dimension mismatch! Expected d_model={self.d_model}, got {result.d_model}."
            )

        if check_nan_inf and hidden_states.numel() > 0:
            if torch.isnan(hidden_states).any().item():
                result.is_valid = False
                result.has_nan = True
                result.errors.append("Detected NaN values in input hidden_states!")

            if torch.isinf(hidden_states).any().item():
                result.is_valid = False
                result.has_inf = True
                result.errors.append("Detected Inf values in input hidden_states!")

        if not result.is_valid:
            logger.error("LM Head input validation failed: %s", result.errors)

        return result
