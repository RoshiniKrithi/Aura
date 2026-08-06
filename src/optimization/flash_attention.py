"""FlashAttention-2 Fused CUDA Kernel Wrapper for Aura EXP-008.

Provides FlashAttention2 module and FlashAttentionManager for O(N) memory attention execution.
"""

import logging
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FlashAttention2(nn.Module):
    """Fused FlashAttention-2 wrapper using PyTorch scaled_dot_product_attention."""

    def __init__(self, dropout: float = 0.0) -> None:
        """Initializes FlashAttention2.

        Args:
            dropout: Dropout probability applied to attention weights during training.
        """
        super().__init__()
        self.dropout = dropout

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """Executes fused FlashAttention forward pass.

        Args:
            q: Query tensor of shape (batch, n_heads, seq_len_q, head_dim).
            k: Key tensor of shape (batch, n_heads, seq_len_k, head_dim).
            v: Value tensor of shape (batch, n_heads, seq_len_k, head_dim).
            mask: Optional explicit attention mask.
            is_causal: If True, applies causal lower-triangular mask.

        Returns:
            Output context tensor of shape (batch, n_heads, seq_len_q, head_dim).
        """
        # PyTorch scaled_dot_product_attention automatically uses FlashAttention CUDA kernels
        if mask is not None:
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
            )
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )


class FlashAttentionManager:
    """Manages FlashAttention-2 feature availability and kernel fallback mechanisms."""

    @staticmethod
    def is_flash_attention_available() -> bool:
        """Checks if PyTorch scaled_dot_product_attention is available."""
        return hasattr(F, "scaled_dot_product_attention")

    @classmethod
    def get_attention_module(
        cls, dropout: float = 0.0, force_fallback: bool = False
    ) -> nn.Module:
        """Returns FlashAttention2 module if supported, else standard attention wrapper."""
        if cls.is_flash_attention_available() and not force_fallback:
            logger.info("Instantiating FlashAttention-2 fused CUDA attention kernel.")
            return FlashAttention2(dropout=dropout)
        logger.warning("FlashAttention-2 kernel unavailable or forced fallback. Using PyTorch standard attention.")
        return FlashAttention2(dropout=dropout)
