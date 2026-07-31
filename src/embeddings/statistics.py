"""Embedding Layer Statistics Extractor for Aura LLM Pipeline.

Computes L2 vector norm distributions, weight matrix statistics, gradient norms,
and vocabulary vector utilization metrics.
"""

from dataclasses import dataclass, field
import logging
from typing import Dict, Union
import torch

from src.embeddings.embedding_layer import EmbeddingLayer

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingStats:
    """Summary container holding quantitative embedding weight and gradient metrics."""

    vocab_size: int
    d_model: int
    mean_weight: float
    std_weight: float
    min_weight: float
    max_weight: float
    mean_vector_l2_norm: float
    min_vector_l2_norm: float
    max_vector_l2_norm: float
    grad_norm: float = 0.0
    zero_vector_tokens: int = 0


class EmbeddingStatistics:
    """Computes statistical metrics over an EmbeddingLayer parameter matrix.

    Time Complexity:
        O(V * d) reduction operations.

    Space Complexity:
        O(V) memory space for vector L2 norms.
    """

    @staticmethod
    def compute_stats(layer: EmbeddingLayer) -> EmbeddingStats:
        """Calculates comprehensive weight and gradient metrics for given EmbeddingLayer.

        Args:
            layer: Instantiated EmbeddingLayer module.

        Returns:
            EmbeddingStats summary object.
        """
        weight = layer.weight.detach().cpu()
        vocab_size, d_model = weight.shape

        mean_w = weight.mean().item()
        std_w = weight.std().item()
        min_w = weight.min().item()
        max_w = weight.max().item()

        # Vector L2 norms per token: shape (V,)
        l2_norms = torch.norm(weight, p=2, dim=1)
        mean_norm = l2_norms.mean().item()
        min_norm = l2_norms.min().item()
        max_norm = l2_norms.max().item()
        zero_vectors = (l2_norms == 0.0).sum().item()

        # Gradient norm check
        grad_norm = 0.0
        if layer.weight.grad is not None:
            grad_norm = torch.norm(layer.weight.grad.detach().cpu(), p=2).item()

        return EmbeddingStats(
            vocab_size=vocab_size,
            d_model=d_model,
            mean_weight=round(mean_w, 6),
            std_weight=round(std_w, 6),
            min_weight=round(min_w, 6),
            max_weight=round(max_w, 6),
            mean_vector_l2_norm=round(mean_norm, 6),
            min_vector_l2_norm=round(min_norm, 6),
            max_vector_l2_norm=round(max_norm, 6),
            grad_norm=round(grad_norm, 6),
            zero_vector_tokens=int(zero_vectors),
        )
