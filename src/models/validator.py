"""AuraGPT Input and Parameter Validator for Aura Architecture.

Enforces 2D token ID shape constraints (B, T), vocabulary bounds, sequence length bounds,
target token shape matching, and numerical stability detection (NaN / Inf).
"""

from dataclasses import dataclass, field
import logging
from typing import List, Optional
import torch

from src.models.exceptions import ModelValidationError

logger = logging.getLogger(__name__)


@dataclass
class ModelValidationResult:
    """Summary diagnostic report output from AuraGPT validation checks."""

    is_valid: bool
    batch_size: int = 0
    sequence_length: int = 0
    has_nan: bool = False
    has_inf: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ModelValidator:
    """Validates input token tensors and targets for AuraGPT.

    Time Complexity:
        O(1) shape checks; O(B * T) min/max vocabulary range scans.

    Space Complexity:
        O(1) scalar memory overhead.
    """

    def __init__(self, vocab_size: int = 50257, max_sequence_length: int = 2048) -> None:
        """Initializes validator with target vocabulary size and context window length.

        Args:
            vocab_size: Model vocabulary size.
            max_sequence_length: Maximum allowed sequence length T.
        """
        if vocab_size <= 0:
            raise ModelValidationError(f"vocab_size must be positive, got {vocab_size}")
        if max_sequence_length <= 0:
            raise ModelValidationError(
                f"max_sequence_length must be positive, got {max_sequence_length}"
            )

        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length

    def validate_inputs(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        check_bounds: bool = True,
    ) -> ModelValidationResult:
        """Validates input token ID tensor input_ids and target token ID tensor targets.

        Args:
            input_ids: Input token ID LongTensor of shape (B, T) or (T,).
            targets: Optional ground-truth target token ID LongTensor of matching shape.
            check_bounds: If True, checks that token IDs are within [0, vocab_size - 1].

        Returns:
            ModelValidationResult summary object.
        """
        result = ModelValidationResult(is_valid=True)

        if not isinstance(input_ids, torch.Tensor):
            result.is_valid = False
            result.errors.append(f"input_ids must be a PyTorch Tensor, got {type(input_ids).__name__}")
            return result

        if input_ids.ndim not in (1, 2):
            result.is_valid = False
            result.errors.append(
                f"input_ids must be 1D (T,) or 2D (B, T), got shape {tuple(input_ids.shape)}"
            )
            return result

        if input_ids.numel() == 0:
            result.is_valid = False
            result.errors.append("Empty input_ids tensor received.")
            return result

        if input_ids.ndim == 2:
            result.batch_size, result.sequence_length = input_ids.shape
        else:
            result.batch_size = 1
            result.sequence_length = input_ids.shape[0]

        if result.sequence_length > self.max_sequence_length:
            result.is_valid = False
            result.errors.append(
                f"Sequence length ({result.sequence_length}) exceeds max context window ({self.max_sequence_length})."
            )

        if check_bounds and input_ids.numel() > 0:
            min_id = input_ids.min().item()
            max_id = input_ids.max().item()

            if min_id < 0 or max_id >= self.vocab_size:
                result.is_valid = False
                result.errors.append(
                    f"Token ID out of bounds! Found IDs in range [{min_id}, {max_id}], expected range [0, {self.vocab_size - 1}]."
                )

        if targets is not None:
            if not isinstance(targets, torch.Tensor):
                result.is_valid = False
                result.errors.append(f"targets must be a PyTorch Tensor, got {type(targets).__name__}")
            elif targets.shape != input_ids.shape:
                result.is_valid = False
                result.errors.append(
                    f"Shape mismatch! input_ids shape {tuple(input_ids.shape)} does not match targets shape {tuple(targets.shape)}."
                )

        if not result.is_valid:
            logger.error("AuraGPT model input validation failed: %s", result.errors)

        return result
