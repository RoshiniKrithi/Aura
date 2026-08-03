"""Production-Grade Language Modeling Head Implementation for Aura LLM Architecture.

Projects d_model-dimensional hidden states into unnormalized vocabulary logits (B, T, vocab_size).
Supports weight tying with token embedding weights, custom bias vectors, and mixed precision.
"""

import logging
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.exceptions import LMHeadValidationError
from src.models.lm_head_config import LMHeadConfig
from src.models.lm_head_validator import LMHeadValidator

logger = logging.getLogger(__name__)


class LanguageModelingHead(nn.Module):
    """Production-grade Language Modeling Projection Head.

    Design Decisions:
        - Maps final Transformer hidden states H (B, T, d_model) to raw vocabulary logits L (B, T, vocab_size).
        - Excludes Softmax from forward pass to allow fused PyTorch cross-entropy loss and flexible inference sampling.
        - Supports weight tying by binding projection weight matrix to input token embedding parameter.

    Time Complexity:
        O(B * T * d_model * vocab_size) matrix multiplication.

    Space Complexity:
        O(vocab_size * d_model) parameter allocation (or 0 additional parameters when weight tying is enabled),
        O(B * T * vocab_size) output tensor allocation.
    """

    def __init__(
        self,
        config: Optional[LMHeadConfig] = None,
        tied_weight: Optional[nn.Parameter] = None,
    ) -> None:
        """Initializes LanguageModelingHead.

        Args:
            config: Optional LMHeadConfig hyperparameter object.
            tied_weight: Optional nn.Parameter weight reference for weight tying.
        """
        super().__init__()

        cfg = config or LMHeadConfig()

        self.config = cfg
        self.d_model = cfg.d_model
        self.vocab_size = cfg.vocab_size
        self.tie_weights = cfg.tie_weights

        # 1. Parameter Creation or Weight Tying Binding
        if self.tie_weights and tied_weight is not None:
            self.weight = tied_weight
        else:
            self.weight = nn.Parameter(torch.empty(self.vocab_size, self.d_model))
            nn.init.normal_(self.weight, mean=0.0, std=0.02)

        # 2. Bias Parameter Setup
        if cfg.bias:
            self.bias = nn.Parameter(torch.zeros(self.vocab_size))
        else:
            self.register_parameter("bias", None)

        # 3. Input Validator
        self.validator = LMHeadValidator(d_model=self.d_model, vocab_size=self.vocab_size)

        logger.info(
            "Instantiated LanguageModelingHead: d_model=%d, vocab_size=%d, tie_weights=%s, bias=%s",
            self.d_model,
            self.vocab_size,
            self.tie_weights,
            cfg.bias,
        )

    def set_tied_weight(self, tied_weight: nn.Parameter) -> None:
        """Binds an external parameter reference (e.g. EmbeddingLayer.weight) for weight tying.

        Args:
            tied_weight: PyTorch nn.Parameter reference of shape (vocab_size, d_model).
        """
        self.weight = tied_weight
        self.tie_weights = True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass projecting hidden states to vocabulary logits.

        Args:
            hidden_states: Input FloatTensor of shape (B, T, d_model) or (T, d_model).

        Returns:
            Logits FloatTensor of shape (B, T, vocab_size) or (T, vocab_size).

        Raises:
            LMHeadValidationError: If input validation checks fail.
        """
        val_res = self.validator.validate_hidden_states(hidden_states)
        if not val_res.is_valid:
            raise LMHeadValidationError(f"LM Head input validation failed: {val_res.errors}")

        # Linear matrix projection: L = H @ W^T (+ b)
        logits = F.linear(hidden_states, self.weight, self.bias)

        return logits

    def extra_repr(self) -> str:
        """Provides readable PyTorch module representation."""
        return (
            f"d_model={self.d_model}, vocab_size={self.vocab_size}, "
            f"tie_weights={self.tie_weights}, bias={self.bias is not None}"
        )


# Class Alias for convenience
LanguageModelHead = LanguageModelingHead

