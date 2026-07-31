"""Single-Head Causal Self-Attention Implementation for Aura LLM Architecture.

Implements pure PyTorch manual query, key, value projections, scaled dot-product attention,
lower-triangular causal masking, softmax probability normalization, and output projection.
"""

import math
import logging
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.attention.config import AttentionConfig
from src.attention.exceptions import AttentionValidationError
from src.attention.mask import AttentionMask
from src.attention.validator import AttentionValidator

logger = logging.getLogger(__name__)


class SelfAttention(nn.Module):
    """Production-grade Single-Head Causal Self-Attention module.

    Design Decisions:
        - Manual linear projections W_q, W_k, W_v, W_o without prebuilt attention abstractions.
        - Strict scaling by 1 / sqrt(d_k) to prevent vanishing gradients in softmax.
        - Lower-triangular causal masking to enforce autoregressive generation (GPT standard).
        - Saves last attention weights tensor (self.last_attention_weights) for analysis.

    Time Complexity:
        O(B * T^2 * d_k + B * T * d_k * d_model) linear projections and matrix multiplications.

    Space Complexity:
        O(B * T^2) attention matrix memory,
        O(d_model * d_k) projection parameters.
    """

    def __init__(
        self,
        config: Optional[AttentionConfig] = None,
        d_model: Optional[int] = None,
        d_attn: Optional[int] = None,
        dropout: float = 0.1,
        causal: bool = True,
        bias: bool = True,
        scale: Optional[float] = None,
    ) -> None:
        """Initializes SelfAttention module.

        Args:
            config: Optional AttentionConfig dataclass.
            d_model: Input/output feature dimension d_model.
            d_attn: Query, Key, Value projection dimension d_k (defaults to d_model).
            dropout: Attention weights dropout probability p.
            causal: If True, applies lower-triangular causal mask.
            bias: If True, enables bias vectors in linear projections.
            scale: Custom scaling factor (defaults to 1 / sqrt(d_k)).
        """
        super().__init__()

        cfg = config or AttentionConfig()

        self.d_model = d_model if d_model is not None else cfg.d_model
        self.d_attn = d_attn if d_attn is not None else cfg.d_attn
        self.dropout_p = dropout if config is None else cfg.dropout
        self.causal = causal if config is None else cfg.causal
        self.bias = bias if config is None else cfg.bias

        # Scale factor: 1 / sqrt(d_k)
        if scale is not None:
            self.scale = scale
        elif cfg.scale is not None:
            self.scale = cfg.scale
        else:
            self.scale = 1.0 / math.sqrt(self.d_attn)

        # 1. Learnable Projection Matrices
        self.q_proj = nn.Linear(self.d_model, self.d_attn, bias=self.bias)
        self.k_proj = nn.Linear(self.d_model, self.d_attn, bias=self.bias)
        self.v_proj = nn.Linear(self.d_model, self.d_attn, bias=self.bias)
        self.out_proj = nn.Linear(self.d_attn, self.d_model, bias=self.bias)

        # 2. Dropout Layer
        self.attn_dropout = nn.Dropout(p=self.dropout_p)

        # 3. Input Validator
        self.validator = AttentionValidator(d_model=self.d_model)

        # Cache for last attention weights (used for statistics and visualization)
        self.last_attention_weights: Optional[torch.Tensor] = None

        logger.info(
            "Instantiated SelfAttention: d_model=%d, d_attn=%d, Causal=%s, Scale=%.4f",
            self.d_model,
            self.d_attn,
            self.causal,
            self.scale,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass computing Single-Head Causal Self-Attention.

        Args:
            x: Input FloatTensor of shape (B, T, d_model) or (T, d_model).
            mask: Optional external mask FloatTensor.

        Returns:
            Self-Attention output FloatTensor of shape (B, T, d_model) or (T, d_model).

        Raises:
            AttentionValidationError: If input validation checks fail.
        """
        # Validate Input Integrity
        val_res = self.validator.validate_input_embeddings(x)
        if not val_res.is_valid:
            raise AttentionValidationError(f"Attention input validation failed: {val_res.errors}")

        # Handle 2D (T, d) vs 3D (B, T, d) inputs
        is_2d = x.ndim == 2
        if is_2d:
            x_in = x.unsqueeze(0)  # Convert (T, d) -> (1, T, d)
        else:
            x_in = x

        # 1. Projections: Q = X @ W_q, K = X @ W_k, V = X @ W_v
        # Shapes: Q, K, V -> (B, T, d_attn)
        q = self.q_proj(x_in)
        k = self.k_proj(x_in)
        v = self.v_proj(x_in)

        # 2. Scaled Dot-Product Attention Scores: S = (Q @ K^T) * scale
        # Q: (B, T, d_attn), K^T: (B, d_attn, T) -> S: (B, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 3. Apply Causal Masking (if causal=True)
        if self.causal:
            scores = AttentionMask.apply_causal_mask(scores, custom_mask=mask)
        elif mask is not None:
            scores = scores + mask

        # 4. Softmax Probability Normalization: A_weights = softmax(scores, dim=-1)
        attn_weights = F.softmax(scores, dim=-1)

        # Cache last attention weights for inspection
        self.last_attention_weights = attn_weights.detach()

        # 5. Dropout
        attn_drop = self.attn_dropout(attn_weights)

        # 6. Weighted Value Aggregation: O_attn = A_drop @ V
        # A_drop: (B, T, T), V: (B, T, d_attn) -> O_attn: (B, T, d_attn)
        attn_out = torch.matmul(attn_drop, v)

        # 7. Output Projection: Y = O_attn @ W_o
        # O_attn: (B, T, d_attn) -> output: (B, T, d_model)
        output = self.out_proj(attn_out)

        if is_2d:
            output = output.squeeze(0)  # Convert (1, T, d) back to (T, d)

        return output

    def extra_repr(self) -> str:
        """Provides readable PyTorch module representation."""
        return (
            f"d_model={self.d_model}, d_attn={self.d_attn}, "
            f"causal={self.causal}, scale={self.scale:.4f}, bias={self.bias}"
        )
