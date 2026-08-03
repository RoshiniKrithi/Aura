"""Loss Input Tensor Validator for Aura Architecture.

Enforces 3D logits shape (B, T, V), 2D target token shape (B, T), sequence dimension matching,
and detects NaN / Inf numerical loss instabilities.
"""

from dataclasses import dataclass, field
import logging
from typing import List
import torch

from src.losses.exceptions import LossValidationError

logger = logging.getLogger(__name__)


@dataclass
class LossValidationResult:
    """Summary diagnostic report output from Loss validation checks."""

    is_valid: bool
    batch_size: int = 0
    sequence_length: int = 0
    vocab_size: int = 0
    has_nan: bool = False
    has_inf: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class LossValidator:
    """Validates input logits and target token tensors for CrossEntropyLoss.

    Time Complexity:
        O(1) shape checks; O(B * T * V) when checking for NaN/Inf.

    Space Complexity:
        O(1) scalar memory overhead.
    """

    def validate_inputs(
        self, logits: torch.Tensor, targets: torch.Tensor, check_nan_inf: bool = True
    ) -> LossValidationResult:
        """Validates logits FloatTensor and targets LongTensor.

        Args:
            logits: Logits FloatTensor of shape (B, T, V) or (T, V).
            targets: Targets LongTensor of shape (B, T) or (T,).
            check_nan_inf: If True, checks for NaN and Inf values in logits.

        Returns:
            LossValidationResult summary object.
        """
        result = LossValidationResult(is_valid=True)

        if not isinstance(logits, torch.Tensor):
            result.is_valid = False
            result.errors.append(f"logits must be a PyTorch Tensor, got {type(logits).__name__}")
            return result

        if not isinstance(targets, torch.Tensor):
            result.is_valid = False
            result.errors.append(f"targets must be a PyTorch Tensor, got {type(targets).__name__}")
            return result

        if logits.ndim not in (2, 3):
            result.is_valid = False
            result.errors.append(
                f"logits must be 2D (T, V) or 3D (B, T, V), got shape {tuple(logits.shape)}"
            )
            return result

        if targets.ndim not in (1, 2):
            result.is_valid = False
            result.errors.append(
                f"targets must be 1D (T,) or 2D (B, T), got shape {tuple(targets.shape)}"
            )
            return result

        if logits.ndim == 3:
            b_l, t_l, result.vocab_size = logits.shape
            result.batch_size = b_l
            result.sequence_length = t_l
        else:
            result.batch_size = 1
            result.sequence_length, result.vocab_size = logits.shape

        if targets.ndim == 2:
            b_t, t_t = targets.shape
        else:
            b_t = 1
            t_t = targets.shape[0]

        if (b_l if logits.ndim == 3 else 1) != b_t or (t_l if logits.ndim == 3 else logits.shape[0]) != t_t:
            result.is_valid = False
            result.errors.append(
                f"Sequence shape mismatch! Logits batch/seq ({b_l if logits.ndim == 3 else 1}, {t_l if logits.ndim == 3 else logits.shape[0]}) does not match Targets ({b_t}, {t_t})."
            )

        if check_nan_inf and logits.numel() > 0:
            if torch.isnan(logits).any().item():
                result.is_valid = False
                result.has_nan = True
                result.errors.append("Detected NaN values in input logits!")

            if torch.isinf(logits).any().item():
                result.is_valid = False
                result.has_inf = True
                result.errors.append("Detected Inf values in input logits!")

        if not result.is_valid:
            logger.error("Loss input validation failed: %s", result.errors)

        return result
