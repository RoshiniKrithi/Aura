"""Decoding & Sampling Strategies for Aura LLM Architecture.

Provides Strategy Pattern abstractions for Greedy Decoding, Temperature Scaling,
Top-K Filtering, Top-P (Nucleus) Sampling, and Composite Pipeline execution.
"""

from abc import ABC, abstractmethod
import logging
from typing import Optional
import torch

from src.inference.config import InferenceConfig

logger = logging.getLogger(__name__)


class BaseSamplingStrategy(ABC):
    """Abstract Base Class for all decoding & logit filtering strategies."""

    @abstractmethod
    def process_logits(self, logits: torch.Tensor, config: InferenceConfig) -> torch.Tensor:
        """Processes raw model logits tensor.

        Args:
            logits: Logit tensor of shape (batch_size, vocab_size).
            config: InferenceConfig hyperparameter object.

        Returns:
            Filtered / scaled logit tensor of shape (batch_size, vocab_size).
        """
        pass


class GreedyStrategy(BaseSamplingStrategy):
    """Greedy Decoding strategy selecting the highest probability token argmax."""

    def process_logits(self, logits: torch.Tensor, config: InferenceConfig) -> torch.Tensor:
        """Returns logit tensor unchanged; argmax token selection will occur downstream."""
        return logits

    @staticmethod
    def select_token(logits: torch.Tensor) -> torch.Tensor:
        """Selects token ID via argmax along vocabulary dimension."""
        return torch.argmax(logits, dim=-1, keepdim=True)


class TemperatureStrategy(BaseSamplingStrategy):
    """Temperature Scaling strategy modulating logit distribution entropy."""

    def process_logits(self, logits: torch.Tensor, config: InferenceConfig) -> torch.Tensor:
        """Scales logits by temperature factor (z / T)."""
        temp = config.temperature
        if temp <= 0.0:
            return logits
        return logits / temp


class TopKStrategy(BaseSamplingStrategy):
    """Top-K Filtering strategy keeping only the top K highest logit candidates."""

    def process_logits(self, logits: torch.Tensor, config: InferenceConfig) -> torch.Tensor:
        """Masks logits outside top-K set to -infinity."""
        k = config.top_k
        if k <= 0 or k >= logits.size(-1):
            return logits

        top_k_values, _ = torch.topk(logits, k=k, dim=-1)
        min_values = top_k_values[:, -1].unsqueeze(-1)
        return torch.where(logits < min_values, torch.tensor(float("-inf"), device=logits.device), logits)


class TopPStrategy(BaseSamplingStrategy):
    """Top-P (Nucleus) Sampling strategy filtering by cumulative probability threshold."""

    def process_logits(self, logits: torch.Tensor, config: InferenceConfig) -> torch.Tensor:
        """Masks logits outside cumulative probability threshold P to -infinity."""
        p = config.top_p
        if p >= 1.0 or p <= 0.0:
            return logits

        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens with cumulative probability above threshold p
        sorted_indices_to_remove = cumulative_probs > p
        # Shift mask right to keep first token exceeding threshold p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        return logits.masked_fill(indices_to_remove, float("-inf"))


class CompositeSamplingStrategy:
    """Chains Temperature, Top-K, and Top-P strategies in sequence."""

    def __init__(self) -> None:
        """Initializes CompositeSamplingStrategy with standard filter pipeline."""
        self.temp_strategy = TemperatureStrategy()
        self.top_k_strategy = TopKStrategy()
        self.top_p_strategy = TopPStrategy()

    def sample_next_token(self, logits: torch.Tensor, config: InferenceConfig) -> torch.Tensor:
        """Applies filter pipeline and samples next token ID.

        Args:
            logits: Unnormalized logits tensor (batch_size, vocab_size).
            config: InferenceConfig parameters.

        Returns:
            Sampled token ID tensor (batch_size, 1).
        """
        if not config.do_sample or config.temperature <= 0.0:
            return GreedyStrategy.select_token(logits)

        # 1. Apply Temperature Scaling
        logits = self.temp_strategy.process_logits(logits, config)

        # 2. Apply Top-K Filtering
        logits = self.top_k_strategy.process_logits(logits, config)

        # 3. Apply Top-P Nucleus Filtering
        logits = self.top_p_strategy.process_logits(logits, config)

        # 4. Softmax and Categorical Sample
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token
