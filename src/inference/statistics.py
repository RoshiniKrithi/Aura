"""Inference Statistics Summary for Aura LLM Architecture.

Computes metrics over text generation runs (generated token count, duration seconds, tokens per second).
"""

from dataclasses import dataclass
import logging
from typing import Dict

logger = logging.getLogger(__name__)


@dataclass
class InferenceStats:
    """Summary container holding generation performance metrics."""

    prompt_token_count: int
    generated_token_count: int
    total_token_count: int
    duration_seconds: float
    tokens_per_second: float


class InferenceStatistics:
    """Calculates generation performance statistics."""

    @staticmethod
    def compute_stats(
        prompt_len: int, generated_len: int, duration_seconds: float
    ) -> InferenceStats:
        """Computes summary generation statistics.

        Args:
            prompt_len: Prompt token count.
            generated_len: Generated new token count.
            duration_seconds: Generation elapsed seconds.

        Returns:
            InferenceStats metrics object.
        """
        total_len = prompt_len + generated_len
        tps = round(generated_len / max(1e-5, duration_seconds), 2)

        return InferenceStats(
            prompt_token_count=prompt_len,
            generated_token_count=generated_len,
            total_token_count=total_len,
            duration_seconds=round(duration_seconds, 2),
            tokens_per_second=tps,
        )
