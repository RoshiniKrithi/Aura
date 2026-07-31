"""Production-Grade Transformer Feed Forward Network (MLP) Implementation for Aura LLM Architecture.

Implements point-wise two-layer linear expansion, non-linear activation (GELU, ReLU, SiLU, SwiGLU),
dropout regularization, and down-projection matching GPT-2, GPT-3, LLaMA, DeepSeek, and Mistral standards.
"""

import logging
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn

from src.ffn.config import FeedForwardConfig
from src.ffn.exceptions import FeedForwardValidationError
from src.ffn.initializer import FeedForwardInitializer
from src.ffn.utilities import FeedForwardUtilities
from src.ffn.validator import FeedForwardValidator

logger = logging.getLogger(__name__)


class FeedForwardNetwork(nn.Module):
    """Production-grade Feed Forward Network (MLP) for Transformer Blocks.

    Design Decisions:
        - 2-layer linear projection (d_model -> d_ff -> d_model) with 4x feature dimension expansion.
        - Supports GELU (GPT standard), ReLU, SiLU, and SwiGLU (LLaMA/DeepSeek standard).
        - Configurable weight initialization strategy.
        - Caches last hidden activations (self.last_hidden_activations) for inspection and statistics.

    Time Complexity:
        O(B * T * d_model * d_ff) linear expansion and down-projection operations.

    Space Complexity:
        O(B * T * d_ff) activation memory overhead,
        O(2 * d_model * d_ff) parameter weights.
    """

    def __init__(
        self,
        config: Optional[FeedForwardConfig] = None,
        d_model: Optional[int] = None,
        d_ff: Optional[int] = None,
        activation: str = "gelu",
        dropout: float = 0.1,
        bias: bool = True,
    ) -> None:
        """Initializes FeedForwardNetwork module.

        Args:
            config: Optional FeedForwardConfig dataclass.
            d_model: Feature dimension d_model.
            d_ff: Hidden expansion dimension d_ff (defaults to 4 * d_model).
            activation: Activation function choice ("gelu", "relu", "silu", "swiglu").
            dropout: Residual dropout probability.
            bias: If True, enables bias terms in linear projections.
        """
        super().__init__()

        cfg = config or FeedForwardConfig()

        self.d_model = d_model if d_model is not None else cfg.d_model
        self.activation_name = activation if config is None else cfg.activation.lower()
        self.dropout_p = dropout if config is None else cfg.dropout
        self.bias = bias if config is None else cfg.bias

        if d_ff is not None:
            self.hidden_dim = d_ff
        else:
            self.hidden_dim = cfg.hidden_dim

        self.is_swiglu = self.activation_name == "swiglu"

        # 1. Projections Construction
        if self.is_swiglu:
            # SwiGLU requires 3 linear layers: W_gate, W_up, W_down
            self.w_gate = nn.Linear(self.d_model, self.hidden_dim, bias=self.bias)
            self.w_up = nn.Linear(self.d_model, self.hidden_dim, bias=self.bias)
            self.w_down = nn.Linear(self.hidden_dim, self.d_model, bias=self.bias)
            self.act_fn = FeedForwardUtilities.get_activation_fn("silu")
        else:
            # Standard GPT FFN: 2 linear layers: W_1 (expansion) and W_2 (projection)
            self.w_1 = nn.Linear(self.d_model, self.hidden_dim, bias=self.bias)
            self.w_2 = nn.Linear(self.hidden_dim, self.d_model, bias=self.bias)
            self.act_fn = FeedForwardUtilities.get_activation_fn(self.activation_name)

        # 2. Dropout Layer
        self.dropout = nn.Dropout(p=self.dropout_p)

        # 3. Input Validator
        self.validator = FeedForwardValidator(d_model=self.d_model)

        # 4. Weight Initialization
        FeedForwardInitializer.initialize_weights(self, cfg)

        # Cache for last hidden activations
        self.last_hidden_activations: Optional[torch.Tensor] = None

        logger.info(
            "Instantiated FeedForwardNetwork: d_model=%d, hidden_dim=%d, Activation='%s', Dropout=%.2f",
            self.d_model,
            self.hidden_dim,
            self.activation_name,
            self.dropout_p,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass executing Feed Forward Network.

        Args:
            x: Input FloatTensor of shape (B, T, d_model) or (T, d_model).

        Returns:
            Output FloatTensor of shape (B, T, d_model) or (T, d_model).

        Raises:
            FeedForwardValidationError: If input validation checks fail.
        """
        val_res = self.validator.validate_input_embeddings(x)
        if not val_res.is_valid:
            raise FeedForwardValidationError(f"FFN input validation failed: {val_res.errors}")

        is_2d = x.ndim == 2
        if is_2d:
            x_in = x.unsqueeze(0)  # (T, d) -> (1, T, d)
        else:
            x_in = x

        if self.is_swiglu:
            # SwiGLU(X) = Dropout(SiLU(X W_gate) * (X W_up)) W_down
            gate = self.act_fn(self.w_gate(x_in))
            up = self.w_up(x_in)
            hidden_act = gate * up
            self.last_hidden_activations = hidden_act.detach()

            hidden_drop = self.dropout(hidden_act)
            output = self.w_down(hidden_drop)
        else:
            # Standard FFN: Y = Dropout(Activation(X W_1)) W_2
            h_raw = self.w_1(x_in)
            h_act = self.act_fn(h_raw)
            self.last_hidden_activations = h_act.detach()

            h_drop = self.dropout(h_act)
            output = self.w_2(h_drop)

        if is_2d:
            output = output.squeeze(0)  # (1, T, d) -> (T, d)

        return output

    def extra_repr(self) -> str:
        """Provides readable PyTorch module representation."""
        return (
            f"d_model={self.d_model}, hidden_dim={self.hidden_dim}, "
            f"activation='{self.activation_name}', dropout={self.dropout_p}"
        )
