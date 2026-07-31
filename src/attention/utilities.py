"""Attention Mathematical Utilities for Aura LLM Architecture.

Provides functional scaled dot-product attention calculation and attention map extraction
helpers for downstream inspection and visualization.
"""

import math
import logging
from typing import Optional, Tuple
import torch
import torch.nn.functional as F

from src.attention.mask import AttentionMask

logger = logging.getLogger(__name__)


class AttentionUtilities:
    """Mathematical utility functions for self-attention operations.

    Design Decisions:
        - Pure PyTorch functional implementation matching scaled dot-product formula.
    """

    @staticmethod
    def compute_scaled_dot_product_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        scale: Optional[float] = None,
        dropout_p: float = 0.0,
        is_causal: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates scaled dot-product attention: Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V.

        Args:
            query: Query FloatTensor of shape (B, T, d_k).
            key: Key FloatTensor of shape (B, T, d_k).
            value: Value FloatTensor of shape (B, T, d_v).
            mask: Optional external mask.
            scale: Custom scaling factor (defaults to 1 / sqrt(d_k)).
            dropout_p: Dropout probability p.
            is_causal: If True, applies causal lower-triangular mask.

        Returns:
            Tuple of (output_tensor, attention_weights_matrix).
        """
        d_k = query.size(-1)
        scale_val = scale if scale is not None else (1.0 / math.sqrt(d_k))

        # 1. Raw Scores: S = (Q @ K^T) * scale
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale_val

        # 2. Causal Masking
        if is_causal:
            scores = AttentionMask.apply_causal_mask(scores, custom_mask=mask)
        elif mask is not None:
            scores = scores + mask

        # 3. Softmax
        attn_weights = F.softmax(scores, dim=-1)

        # 4. Dropout
        if dropout_p > 0.0 and query.requires_grad:
            attn_drop = F.dropout(attn_weights, p=dropout_p)
        else:
            attn_drop = attn_weights

        # 5. Weighted Value Aggregation: O = A @ V
        output = torch.matmul(attn_drop, value)

        return output, attn_weights
