"""Attention Output Projection Wrapper Layer for Aura LLM Architecture.

Encapsulates final linear transformation W_o and residual dropout applied to concatenated
attention head representations.
"""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class AttentionOutputProjection(nn.Module):
    """Linear projection and residual dropout wrapper for Multi-Head Attention outputs.

    Design Decisions:
        - Maps concatenated multi-head representations back into model dimension space d_model.
    """

    def __init__(
        self,
        d_model: int = 768,
        dropout: float = 0.1,
        bias: bool = True,
    ) -> None:
        """Initializes AttentionOutputProjection.

        Args:
            d_model: Feature dimension d_model.
            dropout: Residual dropout probability.
            bias: If True, enables bias in linear layer.
        """
        super().__init__()

        self.d_model = d_model
        self.dropout_p = dropout

        self.c_proj = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.resid_dropout = nn.Dropout(p=self.dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass projecting concatenated head vectors: Y = Dropout(X @ W_o).

        Args:
            x: Concatenated multi-head FloatTensor of shape (B, T, d_model).

        Returns:
            Projected FloatTensor of shape (B, T, d_model).
        """
        return self.resid_dropout(self.c_proj(x))
