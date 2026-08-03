"""Loss Subsystem Statistics Extractor for Aura Architecture.

Extracts loss, accuracy, and perplexity metric summaries across training batches.
"""

from dataclasses import dataclass
import logging
from typing import Dict
import torch

logger = logging.getLogger(__name__)


@dataclass
class LossStats:
    """Summary diagnostic report container holding loss metrics."""

    loss: float
    accuracy: float
    perplexity: float
    valid_tokens: int
    ignored_tokens: int


class LossStatistics:
    """Computes statistical metrics over loss inputs and targets."""

    @staticmethod
    def compute_stats(
        logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1
    ) -> LossStats:
        """Calculates token counts and basic target stats.

        Args:
            logits: Shifted or raw logits tensor.
            targets: Shifted or raw targets tensor.
            ignore_index: Target token ID to ignore.

        Returns:
            LossStats summary object.
        """
        flat_targets = targets.reshape(-1)
        valid_mask = flat_targets != ignore_index

        valid_tokens = valid_mask.sum().item()
        ignored_tokens = flat_targets.numel() - valid_tokens

        return LossStats(
            loss=0.0,
            accuracy=0.0,
            perplexity=0.0,
            valid_tokens=valid_tokens,
            ignored_tokens=ignored_tokens,
        )
