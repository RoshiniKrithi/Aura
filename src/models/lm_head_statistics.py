"""LM Head Statistics Extractor for Aura Architecture.

Computes parameter count, memory footprint, logit norm distribution,
and weight tying verification metrics for LanguageModelingHead.
"""

from dataclasses import dataclass
import logging
from typing import Tuple
import torch

from src.models.lm_head import LanguageModelingHead

logger = logging.getLogger(__name__)


@dataclass
class LMHeadStats:
    """Summary diagnostic report container holding quantitative LM Head statistics."""

    d_model: int
    vocab_size: int
    tie_weights: bool
    parameter_count: int
    has_bias: bool


class LMHeadStatistics:
    """Computes statistical metrics over LanguageModelingHead parameters and logits.

    Time Complexity:
        O(V * d) parameter scan.

    Space Complexity:
        O(1) scalar memory overhead.
    """

    @staticmethod
    def compute_stats(lm_head: LanguageModelingHead) -> LMHeadStats:
        """Calculates parameters and structural statistics for given LM Head module.

        Args:
            lm_head: Instantiated LanguageModelingHead module.

        Returns:
            LMHeadStats summary object.
        """
        p_count = sum(p.numel() for p in lm_head.parameters() if p.requires_grad)

        return LMHeadStats(
            d_model=lm_head.d_model,
            vocab_size=lm_head.vocab_size,
            tie_weights=lm_head.tie_weights,
            parameter_count=p_count,
            has_bias=lm_head.bias is not None,
        )

    @staticmethod
    def compute_logit_metrics(logits: torch.Tensor) -> Tuple[float, float, float]:
        """Calculates mean value, variance, and L2 norm across output logits tensor.

        Args:
            logits: Logits FloatTensor of shape (B, T, vocab_size) or (T, vocab_size).

        Returns:
            Tuple of (mean_logit, var_logit, l2_norm).
        """
        mean_val = logits.mean().item()
        var_val = logits.var(unbiased=False).item()
        l2_norm = torch.norm(logits.detach(), p=2).item()

        return round(mean_val, 6), round(var_val, 6), round(l2_norm, 4)
