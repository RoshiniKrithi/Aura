"""Individual Attention Head Abstraction for Aura LLM Architecture.

Provides inspection wrapper for extracting, visualizing, and auditing individual
attention head score maps h in {0, ..., H-1} within MultiHeadAttention.
"""

import logging
from typing import Optional, Tuple
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class AttentionHead:
    """Helper abstraction for accessing individual attention head statistics and maps.

    Design Decisions:
        - Allows inspecting head-specific attention patterns (e.g. syntax tracking, variable lookup)
          without modifying the primary fused CUDA-optimized MHA forward pass.
    """

    @staticmethod
    def extract_head_attention(
        last_attention_weights: torch.Tensor, head_index: int
    ) -> torch.Tensor:
        """Extracts the 2D or 3D attention weight matrix for a specific head index.

        Args:
            last_attention_weights: Multi-head attention weights FloatTensor of shape (B, H, T, T).
            head_index: Index h of target head (0 <= head_index < H).

        Returns:
            Attention map FloatTensor of shape (B, T, T) for selected head.
        """
        if last_attention_weights is None:
            raise ValueError("No cached attention weights found. Run forward pass first.")

        n_heads = last_attention_weights.size(1)
        if head_index < 0 or head_index >= n_heads:
            raise ValueError(f"head_index ({head_index}) out of range [0, {n_heads - 1}].")

        return last_attention_weights[:, head_index, :, :]
