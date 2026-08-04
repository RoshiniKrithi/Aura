"""Transformer Block Input and Parameter Validator for Aura LLM Architecture.

Enforces 2D/3D tensor shape constraints, feature dimension matching d_model,
causal mask shape checking, and numerical stability detection (NaN / Inf).
"""

from dataclasses import dataclass, field
import logging
from typing import List, Optional
import torch

from src.transformer.exceptions import TransformerBlockValidationError

logger = logging.getLogger(__name__)



@dataclass
class TransformerBlockValidationResult:
    """Summary diagnostic report output from TransformerBlock validation checks."""

    is_valid: bool
    batch_size: int = 0
    sequence_length: int = 0
    d_model: int = 0
    has_nan: bool = False
    has_inf: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class TransformerBlockValidator:
    """Validates input tensors and mask shapes for TransformerBlock.

    Time Complexity:
        O(1) shape checks; O(B * T * d) torch.isnan / torch.isinf reduction scans.

    Space Complexity:
        O(1) scalar memory overhead.
    """

    def __init__(self, d_model: int = 768) -> None:
        """Initializes validator with target model dimension.

        Args:
            d_model: Expected input/output feature dimension d_model.
        """
        if d_model <= 0:
            raise TransformerBlockValidationError(f"d_model must be positive, got {d_model}")

        self.d_model = d_model

    def validate_inputs(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        check_values: bool = True,
    ) -> TransformerBlockValidationResult:
        """Validates input tensor x and optional attention mask tensor.

        Args:
            x: Input sequence FloatTensor of shape (B, T, d_model) or (T, d_model).
            attention_mask: Optional mask FloatTensor of shape (B, 1, T, T) or (1, 1, T, T).
            check_values: If True, scans tensor for NaN or Inf entries.

        Returns:
            TransformerBlockValidationResult summary object.
        """
        result = TransformerBlockValidationResult(is_valid=True)

        if not isinstance(x, torch.Tensor):
            result.is_valid = False
            result.errors.append(f"Input x must be a PyTorch Tensor, got {type(x).__name__}")
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

        if feat_dim != self.d_model:
            result.is_valid = False
            result.errors.append(
                f"Feature dimension mismatch! Input feature dim ({feat_dim}) does not match configured d_model ({self.d_model})."
            )

        if attention_mask is not None:
            if not isinstance(attention_mask, torch.Tensor):
                result.is_valid = False
                result.errors.append(
                    f"Attention mask must be a PyTorch Tensor, got {type(attention_mask).__name__}"
                )
            elif attention_mask.ndim not in (2, 3, 4):
                result.is_valid = False
                result.errors.append(
                    f"Attention mask must be 2D, 3D, or 4D, got shape {tuple(attention_mask.shape)}"
                )

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
            logger.error("TransformerBlock input validation failed: %s", result.errors)

        return result
