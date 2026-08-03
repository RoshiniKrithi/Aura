"""Model Utilities for Aura LLM Architecture.

Provides functional helper routines for hidden state inspection, parameter counting,
activation norm metrics, and diagnostic tensor operations.
"""

import logging
from typing import Dict, Tuple
import torch

logger = logging.getLogger(__name__)


class ModelUtilities:
    """Functional utility helper routines for GPTModel and AuraGPT."""

    @staticmethod
    def compute_hidden_state_metrics(hidden_states: torch.Tensor) -> Tuple[float, float, float]:
        """Calculates mean value, mean variance, and L2 norm over hidden states tensor.

        Args:
            hidden_states: FloatTensor of shape (B, T, d_model) or (T, d_model).

        Returns:
            Tuple of (mean_value, mean_variance, l2_norm).
        """
        mean_val = hidden_states.mean().item()
        var_val = hidden_states.var(unbiased=False).item()
        l2_norm = torch.norm(hidden_states.detach(), p=2).item()

        return round(mean_val, 6), round(var_val, 6), round(l2_norm, 4)
