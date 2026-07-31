"""Layer Normalization Utilities for Aura LLM Architecture.

Provides functional layer normalization calculation helper functions for downstream inspection.
"""

import logging
from typing import Optional, Tuple
import torch

logger = logging.getLogger(__name__)


class LayerNormUtilities:
    """Functional utility functions for layer normalization."""

    @staticmethod
    def compute_layer_norm(
        x: torch.Tensor,
        gamma: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
        eps: float = 1e-5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Functional layer normalization: y = gamma * ((x - mean) / sqrt(var + eps)) + beta.

        Args:
            x: Input FloatTensor of shape (B, T, d_model) or (T, d_model).
            gamma: Optional scale FloatTensor of shape (d_model,).
            beta: Optional shift FloatTensor of shape (d_model,).
            eps: Numerical stability constant.

        Returns:
            Tuple of (output_tensor, mean_tensor, variance_tensor).
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        x_hat = (x - mean) / torch.sqrt(var + eps)

        output = x_hat
        if gamma is not None:
            output = output * gamma
        if beta is not None:
            output = output + beta

        return output, mean, var
