"""Residual Connection Input and Parameter Validator for Aura LLM Architecture.

Enforces shape consistency between identity input tensor x and sub-layer output tensor sub_out,
checks feature dimensions, non-empty tensors, and scans for numerical instabilities (NaN / Inf).
"""

from dataclasses import dataclass, field
import logging
from typing import List
import torch

from src.transformer.exceptions import ResidualValidationError

logger = logging.getLogger(__name__)


@dataclass
class ResidualValidationResult:
    """Summary diagnostic report output from Residual validation checks."""

    is_valid: bool
    batch_size: int = 0
    sequence_length: int = 0
    d_model: int = 0
    has_nan: bool = False
    has_inf: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ResidualValidator:
    """Validates input tensors and shapes for ResidualConnection.

    Time Complexity:
        O(1) shape checks; O(B * T * d) torch.isnan / torch.isinf reduction scans.

    Space Complexity:
        O(1) scalar memory overhead.
    """

    def __init__(self, d_model: int = 768, dropout: float = 0.1) -> None:
        """Initializes validator with target model dimension and dropout rate.

        Args:
            d_model: Expected input/output feature dimension d_model.
            dropout: Dropout probability.
        """
        if d_model <= 0:
            raise ResidualValidationError(f"d_model must be positive, got {d_model}")
        if not (0.0 <= dropout < 1.0):
            raise ResidualValidationError(f"dropout must be in [0, 1), got {dropout}")

        self.d_model = d_model
        self.dropout = dropout

    def validate_tensors(
        self, x: torch.Tensor, sub_out: torch.Tensor, check_values: bool = True
    ) -> ResidualValidationResult:
        """Validates identity tensor x and sub-layer output tensor sub_out.

        Args:
            x: Identity input FloatTensor of shape (B, T, d_model) or (T, d_model).
            sub_out: Sub-layer output FloatTensor of matching shape.
            check_values: If True, scans tensors for NaN or Inf entries.

        Returns:
            ResidualValidationResult summary object.
        """
        result = ResidualValidationResult(is_valid=True)

        if not isinstance(x, torch.Tensor) or not isinstance(sub_out, torch.Tensor):
            result.is_valid = False
            result.errors.append("Both x and sub_out must be PyTorch Tensors.")
            return result

        if x.shape != sub_out.shape:
            result.is_valid = False
            result.errors.append(
                f"Shape mismatch! Identity tensor shape {tuple(x.shape)} does not match sub-layer output shape {tuple(sub_out.shape)}."
            )
            return result

        if x.ndim not in (2, 3):
            result.is_valid = False
            result.errors.append(
                f"Input tensor must be 2D (T, d) or 3D (B, T, d), got shape {tuple(x.shape)}"
            )
            return result

        if x.numel() == 0:
            result.is_valid = False
            result.errors.append("Empty tensor received.")
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
                f"Feature dimension mismatch! Tensor feature dim ({feat_dim}) does not match configured d_model ({self.d_model})."
            )

        if check_values:
            if torch.isnan(x).any().item() or torch.isnan(sub_out).any().item():
                result.is_valid = False
                result.has_nan = True
                result.errors.append("Tensors contain NaN values!")

            if torch.isinf(x).any().item() or torch.isinf(sub_out).any().item():
                result.is_valid = False
                result.has_inf = True
                result.errors.append("Tensors contain Inf / -Inf values!")

        if not result.is_valid:
            logger.error("Residual input validation failed: %s", result.errors)

        return result
