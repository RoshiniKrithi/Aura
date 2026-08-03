"""Residual Connection Utilities for Aura LLM Architecture.

Provides functional residual calculation helper functions for downstream inspection and debugging.
"""

import logging
from typing import Optional, Tuple
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ResidualUtilities:
    """Functional utility functions for residual connections."""

    @staticmethod
    def apply_residual(
        x: torch.Tensor,
        sub_out: torch.Tensor,
        dropout_p: float = 0.0,
        training: bool = False,
    ) -> Tuple[torch.Tensor, float, float]:
        """Functional residual addition: y = x + Dropout(sub_out).

        Args:
            x: Input identity FloatTensor of shape (B, T, d_model) or (T, d_model).
            sub_out: Sub-layer output FloatTensor of matching shape.
            dropout_p: Dropout probability.
            training: Whether in training mode for dropout evaluation.

        Returns:
            Tuple of (output_tensor, input_l2_norm, output_l2_norm).
        """
        dropped_sub = (
            nn.functional.dropout(sub_out, p=dropout_p, training=training)
            if dropout_p > 0.0
            else sub_out
        )

        out = x + dropped_sub
        x_norm = torch.norm(x.detach(), p=2).item()
        out_norm = torch.norm(out.detach(), p=2).item()

        return out, x_norm, out_norm
