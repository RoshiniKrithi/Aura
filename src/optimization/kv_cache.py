"""Paged and Pre-allocated Key-Value (KV) Cache Manager for Aura EXP-008.

Provides KVCacheManager for 0ms token step allocation latency during autoregressive decoding.
"""

import logging
from typing import Optional, Tuple
import torch

logger = logging.getLogger(__name__)


class KVCacheManager:
    """Pre-allocated continuous tensor Key-Value Cache manager for fast token decoding."""

    def __init__(
        self,
        batch_size: int = 1,
        n_heads: int = 12,
        max_seq_len: int = 2048,
        head_dim: int = 64,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Initializes KVCacheManager and pre-allocates continuous GPU/CPU RAM tensors.

        Args:
            batch_size: Max batch size capacity.
            n_heads: Number of attention heads.
            max_seq_len: Max context sequence length L.
            head_dim: Per-head vector dimension (d_model / n_heads).
            dtype: Tensor data precision type.
            device: Execution target device.
        """
        self.batch_size = batch_size
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        # Pre-allocate zero tensors (batch, n_heads, max_seq_len, head_dim)
        self.key_cache = torch.zeros(
            (batch_size, n_heads, max_seq_len, head_dim), dtype=dtype, device=device
        )
        self.value_cache = torch.zeros(
            (batch_size, n_heads, max_seq_len, head_dim), dtype=dtype, device=device
        )
        self.current_step = 0

        logger.debug(
            "Pre-allocated KV Cache Tensors: shape=%s, device=%s",
            self.key_cache.shape,
            self.device,
        )

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, step: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Appends new step K and V tensors to pre-allocated cache at step index.

        Args:
            key_states: Tensor of shape (batch, n_heads, seq_len_step, head_dim).
            value_states: Tensor of shape (batch, n_heads, seq_len_step, head_dim).
            step: Active sequence position step index.

        Returns:
            Tuple of (all_keys_slice, all_values_slice) up to step + seq_len_step.
        """
        seq_len_step = key_states.shape[2]
        start_idx = step
        end_idx = step + seq_len_step

        if end_idx > self.max_seq_len:
            raise ValueError(
                f"KV Cache step {end_idx} exceeds maximum pre-allocated sequence length {self.max_seq_len}"
            )

        # Slice update without re-allocation
        self.key_cache[:, :, start_idx:end_idx, :] = key_states
        self.value_cache[:, :, start_idx:end_idx, :] = value_states
        self.current_step = end_idx

        return (
            self.key_cache[:, :, :end_idx, :],
            self.value_cache[:, :, :end_idx, :],
        )

    def reset(self) -> None:
        """Resets step pointer without freeing allocated GPU/CPU memory."""
        self.key_cache.zero_()
        self.value_cache.zero_()
        self.current_step = 0
