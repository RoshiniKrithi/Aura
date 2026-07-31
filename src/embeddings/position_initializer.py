"""Position Embedding Weight Initializer for Aura LLM Architecture.

Initializes learnable position parameter matrix P using PyTorch distributions
such as Normal, Uniform, Xavier, Kaiming, and Truncated Normal.
"""

import logging
import torch
import torch.nn as nn

from src.embeddings.exceptions import EmbeddingInitializationError

logger = logging.getLogger(__name__)


class PositionEmbeddingInitializer:
    """Weight initializer for positional embedding parameter matrices.

    Design Decisions:
        - Pure PyTorch tensor ops without third-party dependencies.
        - Supports GPT-2 style small standard deviation initialization (init_std=0.02).
    """

    @staticmethod
    def initialize(
        weight: torch.nn.Parameter | torch.Tensor,
        method: str = "normal",
        init_range: float = 0.02,
        init_std: float = 0.02,
    ) -> torch.Tensor:
        """Initializes positional weight matrix P in-place.

        Args:
            weight: PyTorch Parameter or Tensor of shape (max_sequence_length, d_model).
            method: Initialization method name.
            init_range: Range for uniform initializations.
            init_std: Standard deviation for Normal and Truncated Normal distributions.

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
                nn.init.normal_(weight, mean=0.0, std=init_std)
                weight.clamp_(-2.0 * init_std, 2.0 * init_std)
            else:
                raise EmbeddingInitializationError(
                    f"Unsupported position embedding initialization method: '{method}'."
                )

        logger.info(
            "Initialized position embedding matrix (%s) using method='%s'",
            tuple(weight.shape),
            method_lower,
        )
        return weight
