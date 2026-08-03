"""Transformer Block Statistics Extractor for Aura LLM Pipeline.

Computes parameter count breakdown across sub-modules (MHA, FFN, LN),
activation norm distributions, and parameter gradient norms.
"""

from dataclasses import dataclass
import logging
import torch

from src.transformer.transformer_block import TransformerBlock

logger = logging.getLogger(__name__)


@dataclass
class TransformerBlockStats:
    """Summary container holding quantitative TransformerBlock statistics."""

    d_model: int
    n_heads: int
    d_ff: int
    total_parameters: int
    attn_parameters: int
    ffn_parameters: int
    ln_parameters: int
    last_attn_norm: float = 0.0
    last_ffn_norm: float = 0.0


class TransformerBlockStatistics:
    """Computes quantitative statistics over TransformerBlock parameters and activations.

    Time Complexity:
        O(P) scan over module parameters.

    Space Complexity:
        O(1) scalar memory overhead.
    """

    @staticmethod
    def compute_stats(block: TransformerBlock) -> TransformerBlockStats:
        """Calculates parameters and operational metrics for given TransformerBlock module.

        Args:
            block: Instantiated TransformerBlock module.

        Returns:
            TransformerBlockStats summary object.
        """
        total_p = sum(p.numel() for p in block.parameters() if p.requires_grad)
        attn_p = sum(p.numel() for p in block.attn.parameters() if p.requires_grad)
        ffn_p = sum(p.numel() for p in block.ffn.parameters() if p.requires_grad)
        ln1_p = sum(p.numel() for p in block.ln_1.parameters() if p.requires_grad)
        ln2_p = sum(p.numel() for p in block.ln_2.parameters() if p.requires_grad)

        attn_norm = block.last_attn_norm if block.last_attn_norm is not None else 0.0
        ffn_norm = block.last_ffn_norm if block.last_ffn_norm is not None else 0.0

        return TransformerBlockStats(
            d_model=block.d_model,
            n_heads=block.n_heads,
            d_ff=block.d_ff,
            total_parameters=total_p,
            attn_parameters=attn_p,
            ffn_parameters=ffn_p,
            ln_parameters=ln1_p + ln2_p,
            last_attn_norm=round(attn_norm, 4),
            last_ffn_norm=round(ffn_norm, 4),
        )
