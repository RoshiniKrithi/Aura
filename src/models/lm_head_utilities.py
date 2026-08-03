"""LM Head Utilities Module for Aura LLM Architecture.

Provides functional utility helper routines for LanguageModelingHead, logit state inspection,
weight tying binding, and top-k logit analysis.
"""

import logging
from typing import Tuple
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LMHeadUtilities:
    """Functional utility helper routines for LanguageModelingHead modules.

    Design Decisions:
        - Stateless, pure functional helper routines.
        - Provides diagnostic metrics over raw output logits.
        - Provides weight tying binding functions.
    """

    @staticmethod
    def apply_weight_tying(lm_head: nn.Module, embedding_weight: nn.Parameter) -> None:
        """Binds token embedding weight matrix to LM Head for weight sharing.

        Args:
            lm_head: Instantiated LanguageModelingHead module.
            embedding_weight: PyTorch Parameter reference from EmbeddingLayer.
        """
        if hasattr(lm_head, "set_tied_weight"):
            lm_head.set_tied_weight(embedding_weight)
        else:
            lm_head.weight = embedding_weight

        logger.info("Bound embedding weight matrix for weight tying in LM Head.")

    @staticmethod
    def compute_logit_stats(logits: torch.Tensor) -> Tuple[float, float, float, float, float]:
        """Calculates mean, variance, L2 norm, min, and max values across logits tensor.

        Args:
            logits: Output FloatTensor of shape (B, T, vocab_size) or (T, vocab_size).

        Returns:
            Tuple of (mean_val, var_val, l2_norm, min_val, max_val).
        """
        mean_val = logits.mean().item()
        var_val = logits.var(unbiased=False).item()
        l2_norm = torch.norm(logits.detach(), p=2).item()
        min_val = logits.min().item()
        max_val = logits.max().item()

        return (
            round(mean_val, 6),
            round(var_val, 6),
            round(l2_norm, 4),
            round(min_val, 4),
            round(max_val, 4),
        )

    @staticmethod
    def inspect_top_k_logits(logits: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieves top-k largest logit values and token indices for the final sequence position.

        Args:
            logits: Logits FloatTensor of shape (B, T, vocab_size) or (T, vocab_size).
            k: Number of top logits to extract (default: 5).

        Returns:
            Tuple of (top_k_values, top_k_indices).
        """
        if logits.ndim == 3:
            last_token_logits = logits[:, -1, :]  # (B, vocab_size)
        elif logits.ndim == 2:
            last_token_logits = logits[-1, :]  # (vocab_size,)
        else:
            last_token_logits = logits

        top_values, top_indices = torch.topk(last_token_logits, k=k, dim=-1)
        return top_values, top_indices
