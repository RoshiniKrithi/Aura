"""Scaled Weight Initializer for AuraGPT Architecture.

Implements normal weight initialization (mean=0.0, std=0.02) and scaled initialization
std = 0.02 / sqrt(2 * n_layers) for residual output projections matching GPT-2/3 standards.
"""

import math
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelInitializer:
    """Weight initialization engine for AuraGPT models."""

    @staticmethod
    def initialize_weights(model: nn.Module, initializer_range: float = 0.02, n_layers: int = 12) -> None:
        """Applies scaled weight initialization across all sub-modules in AuraGPT.

        Args:
            model: Instantiated AuraGPT PyTorch Module.
            initializer_range: Standard deviation for weight initialization (default: 0.02).
            n_layers: Number of stacked Transformer blocks N.
        """
        scaled_std = initializer_range / math.sqrt(2.0 * max(1, n_layers))

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Standard linear projection weight initialization
                if "c_proj" in name or "w_2" in name or "w_down" in name:
                    # Scaled initialization for residual output projection weights
                    nn.init.normal_(module.weight, mean=0.0, std=scaled_std)
                else:
                    nn.init.normal_(module.weight, mean=0.0, std=initializer_range)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=initializer_range)

        logger.info(
            "Completed scaled weight initialization: base_std=%.4f, residual_scaled_std=%.6f, n_layers=%d",
            initializer_range,
            scaled_std,
            n_layers,
        )
