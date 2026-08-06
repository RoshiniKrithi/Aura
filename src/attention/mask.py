"""Causal and Padding Attention Mask Utilities for Aura LLM Architecture.

Generates lower-triangular causal masks and combines padding masks to enforce
autoregressive sequence generation in GPT models.
"""

import logging
from typing import Optional, Union
import torch

logger = logging.getLogger(__name__)


class AttentionMask:
    """Utility engine for constructing and applying attention masks.

    Design Decisions:
        - Uses lower-triangular torch.tril matrix generation for causal masking.
        - Fills masked future positions with -1e9 / -inf to guarantee zero probability after softmax.
        - Supports 2D (T, T) and 3D (B, T, T) mask tensors.

    Time Complexity:
        O(T^2) matrix generation.

    Space Complexity:
        O(T^2) mask tensor allocation.
    """

    @staticmethod
    def create_causal_mask(
        sequence_length: int,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Generates a lower-triangular causal mask matrix of shape (T, T).

        Values:
            - 0.0 at allowed positions (j <= i)
            - -inf (-1e9) at forbidden future positions (j > i)

        Args:
            sequence_length: Sequence length T.
            device: Target execution device.
            dtype: Target floating point dtype (float32, float16, bfloat16).

        Returns:
            Causal mask FloatTensor of shape (sequence_length, sequence_length).
        """
        if sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {sequence_length}")

        target_device = device or "cpu"
        # 1. Lower triangular boolean mask: True at j <= i, False at j > i
        tril_bool = torch.tril(
            torch.ones((sequence_length, sequence_length), dtype=torch.bool, device=target_device)
        )

        # 2. Fill allowed positions with 0.0, forbidden positions with finite negative value
        if dtype == torch.float16:
            fill_val = -65000.0
        elif dtype == torch.bfloat16:
            fill_val = -1e9
        else:
            fill_val = float("-inf")
        mask = torch.zeros((sequence_length, sequence_length), dtype=dtype, device=target_device)
        mask.masked_fill_(~tril_bool, fill_val)

        return mask

    @classmethod
    def apply_causal_mask(
        cls,
        attention_scores: torch.Tensor,
        custom_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Applies causal mask to raw attention scores tensor of shape (T, T), (B, T, T), or (B, H, T, T).

        Args:
            attention_scores: FloatTensor of raw scaled scores (Q @ K^T / sqrt(d_k)).
            custom_mask: Optional external mask tensor (e.g. combined padding mask).

        Returns:
            Masked attention scores FloatTensor of same shape.
        """
        if attention_scores.ndim not in (2, 3, 4):
            raise ValueError(
                f"attention_scores must be 2D (T, T), 3D (B, T, T), or 4D (B, H, T, T), got shape {tuple(attention_scores.shape)}"
            )

        t_q = attention_scores.size(-2)
        t_k = attention_scores.size(-1)

        if t_q == t_k:
            causal_m = cls.create_causal_mask(
                sequence_length=t_k,
                device=attention_scores.device,
                dtype=attention_scores.dtype,
            )

            if attention_scores.ndim == 3 and causal_m.ndim == 2:
                causal_m = causal_m.unsqueeze(0)  # (1, T, T)
            elif attention_scores.ndim == 4 and causal_m.ndim == 2:
                causal_m = causal_m.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

            masked_scores = attention_scores + causal_m
        else:
            # For KV-cache decoding (T_q < T_k), query token attends to all past cached tokens
            masked_scores = attention_scores

        if custom_mask is not None:
            masked_scores = masked_scores + custom_mask

        return masked_scores
