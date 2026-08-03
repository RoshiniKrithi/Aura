"""AuraGPT Model Statistics Extractor for Aura LLM Pipeline.

Computes parameter breakdown (embeddings, blocks, LM head), total parameter count,
non-embedding parameter count, and activation norm distributions across blocks.
"""

from dataclasses import dataclass, field
import logging
from typing import Dict
import torch

from src.models.gpt import AuraGPT

logger = logging.getLogger(__name__)


@dataclass
class ModelStats:
    """Summary container holding quantitative AuraGPT model statistics."""

    model_name: str
    total_parameters: int
    non_embedding_parameters: int
    embedding_parameters: int
    lm_head_parameters: int
    n_layers: int
    d_model: int
    n_heads: int
    vocab_size: int
    tie_weights: bool


class ModelStatistics:
    """Computes statistical metrics over AuraGPT model parameters and activations.

    Time Complexity:
        O(P) scan over model parameter tensors.

    Space Complexity:
        O(1) scalar memory overhead.
    """

    @staticmethod
    def compute_stats(model: AuraGPT) -> ModelStats:
        """Calculates comprehensive metrics for given AuraGPT model.

        Args:
            model: Instantiated AuraGPT model module.

        Returns:
            ModelStats summary object.
        """
        total_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        tok_p = sum(p.numel() for p in model.tok_embeddings.parameters() if p.requires_grad)

        if hasattr(model.pos_embeddings, "parameters"):
            pos_p = sum(p.numel() for p in model.pos_embeddings.parameters() if p.requires_grad)
        else:
            pos_p = 0

        head_p = (
            0
            if model.tie_weights
            else sum(p.numel() for p in model.lm_head.parameters() if p.requires_grad)
        )

        non_emb_p = total_p - (tok_p + pos_p)

        return ModelStats(
            model_name=model.model_name,
            total_parameters=total_p,
            non_embedding_parameters=non_emb_p,
            embedding_parameters=tok_p + pos_p,
            lm_head_parameters=head_p,
            n_layers=model.n_layers,
            d_model=model.d_model,
            n_heads=model.n_heads,
            vocab_size=model.vocab_size,
            tie_weights=model.tie_weights,
        )
