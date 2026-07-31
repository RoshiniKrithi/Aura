"""Production-Grade Layer Normalization Implementation for Aura LLM Architecture.

Implements manual mean calculation, variance calculation, standardization with numerical stability epsilon,
and learnable affine scale (gamma) and shift (beta) parameters matching GPT-2/3 standards.
"""

import logging
from typing import Optional, Union
import torch
import torch.nn as nn

from src.normalization.config import LayerNormConfig
from src.normalization.exceptions import LayerNormValidationError
from src.normalization.validator import LayerNormValidator

logger = logging.getLogger(__name__)


class LayerNormalization(nn.Module):
    """Production-grade Layer Normalization module.

    Design Decisions:
        - Pure PyTorch manual mean and variance tensor reductions over the feature dimension (dim=-1).
        - Includes epsilon (default 1e-5) inside the square root to prevent division by zero.
        - Learnable scale (gamma, initialized to 1.0) and shift (beta, initialized to 0.0).
        - Batch-independent normalization operating per token vector (1, 1, d_model).
        - Caches last mean and variance tensors (self.last_mean, self.last_variance) for analysis.

    Time Complexity:
        O(B * T * d_model) mean, variance, and affine elementwise operations.

    Space Complexity:
        O(B * T * 1) mean/variance memory,
        O(2 * d_model) learnable gamma and beta parameters.
    """

    def __init__(

        self,
        config: Optional[LayerNormConfig] = None,
        d_model: Optional[int] = None,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
    ) -> None:
        """Initializes LayerNormalization module.

        Args:
            config: Optional LayerNormConfig dataclass.
            d_model: Feature dimension d_model.
            eps: Numerical stability constant epsilon.
            elementwise_affine: If True, includes learnable gamma and beta parameters.
            bias: If True, includes learnable beta bias parameter when elementwise_affine=True.
        """
        super().__init__()

        cfg = config or LayerNormConfig()

        self.d_model = d_model if d_model is not None else cfg.d_model
        self.eps = eps if config is None else cfg.eps
        self.elementwise_affine = elementwise_affine if config is None else cfg.elementwise_affine
        self.use_bias = bias if config is None else cfg.bias

        # 1. Learnable Affine Parameters: Gamma (Scale) and Beta (Shift)
        if self.elementwise_affine:
            self.gamma = nn.Parameter(torch.ones(self.d_model))
            if self.use_bias:
                self.beta = nn.Parameter(torch.zeros(self.d_model))
            else:
                self.register_parameter("beta", None)
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

        # 2. Input Validator
        self.validator = LayerNormValidator(d_model=self.d_model, eps=self.eps)

        # Cache for last mean and variance tensors
        self.last_mean: Optional[torch.Tensor] = None
        self.last_variance: Optional[torch.Tensor] = None

        logger.info(
            "Instantiated LayerNormalization: d_model=%d, eps=%e, Affine=%s",
            self.d_model,
            self.eps,
            self.elementwise_affine,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computing Layer Normalization.

        Args:
            x: Input FloatTensor of shape (B, T, d_model) or (T, d_model).

        Returns:
            Normalized FloatTensor of shape (B, T, d_model) or (T, d_model).

        Raises:
            LayerNormValidationError: If input validation checks fail.
        """
        # Validate Input Integrity
        val_res = self.validator.validate_input_embeddings(x)
        if not val_res.is_valid:
            raise LayerNormValidationError(f"LayerNorm input validation failed: {val_res.errors}")

        # 1. Compute Mean across last dimension (keepdim=True for broadcasting)
        # Shape: (B, T, 1) or (T, 1)
        mean = x.mean(dim=-1, keepdim=True)

        # 2. Compute Variance across last dimension (unbiased=False for population variance)
        # Shape: (B, T, 1) or (T, 1)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)

        # Cache mean and variance for inspection
        self.last_mean = mean.detach()
        self.last_variance = variance.detach()

        # 3. Standardize: x_hat = (x - mean) / sqrt(variance + eps)
        # Shape: (B, T, d_model) or (T, d_model)
        x_hat = (x - mean) / torch.sqrt(variance + self.eps)

        # 4. Scale (Gamma) and Shift (Beta)
        output = x_hat
        if self.gamma is not None:
            output = output * self.gamma
        if self.beta is not None:
            output = output + self.beta

        return output

    def extra_repr(self) -> str:
        """Provides readable PyTorch module representation."""
        return (
            f"d_model={self.d_model}, eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}"
        )
