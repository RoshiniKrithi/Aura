"""Weight Initialization Engine for Aura Embedding Layer.

Implements modular weight initializations including Normal, Uniform, Xavier,
Kaiming, and Truncated Normal distributions.
"""

import logging
from typing import Optional
import torch
import torch.nn as nn

from src.embeddings.exceptions import EmbeddingInitializationError

logger = logging.getLogger(__name__)


class EmbeddingInitializer:
    """Configurable weight initializer for PyTorch embedding parameter tensors.

    Design Decisions:
        - Pure PyTorch tensor operations without third-party dependencies.
        - Supports pad_idx zero-initialization to ensure pad token vector remains zero.
    """

    @staticmethod
    def initialize(
        weight: torch.nn.Parameter | torch.Tensor,
        method: str = "normal",
        init_range: float = 0.02,
        init_std: float = 0.02,
        pad_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Initializes target embedding weight tensor in-place.

        Args:
            weight: PyTorch Parameter or Tensor of shape (vocab_size, d_model).
            method: Initialization method name.
            init_range: Uniform distribution half-width [-init_range, init_range].
            init_std: Standard deviation for Normal / Truncated Normal distributions.
            pad_idx: Optional token index to zero out post-initialization.

        Returns:
            Initialized PyTorch Tensor reference.

        Raises:
            EmbeddingInitializationError: If an unsupported initialization method is requested.
        """
        method_lower = method.lower().strip()

        with torch.no_grad():
            if method_lower in ("normal", "gaussian"):
                nn.init.normal_(weight, mean=0.0, std=init_std)
            elif method_lower in ("uniform",):
                nn.init.uniform_(weight, a=-init_range, b=init_range)
            elif method_lower in ("xavier_uniform", "glorot_uniform"):
                nn.init.xavier_uniform_(weight)
            elif method_lower in ("xavier_normal", "glorot_normal"):
                nn.init.xavier_normal_(weight)
            elif method_lower in ("kaiming_uniform", "he_uniform"):
                nn.init.kaiming_uniform_(weight, a=0, nonlinearity="linear")
            elif method_lower in ("truncated_normal", "trunc_normal"):
                # Truncated normal distribution within 2 * init_std bounds
                nn.init.normal_(weight, mean=0.0, std=init_std)
                weight.clamp_(-2.0 * init_std, 2.0 * init_std)
            else:
                raise EmbeddingInitializationError(
                    f"Unsupported embedding initialization method: '{method}'. "
                    f"Supported choices: ['normal', 'uniform', 'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'truncated_normal']."
                )

            # Zero out padding index vector if pad_idx is specified
            if pad_idx is not None and 0 <= pad_idx < weight.size(0):
                weight[pad_idx].zero_()

        logger.info(
            "Initialized embedding matrix (%s) using method='%s' (pad_idx=%s)",
            tuple(weight.shape),
            method_lower,
            pad_idx,
        )
        return weight
