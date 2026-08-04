"""Metric Tracker Accumulator for Aura LLM Architecture.

Accumulates training and evaluation metrics (loss, accuracy, perplexity, learning rate, grad norm)
and calculates epoch summaries.
"""

from dataclasses import dataclass, field
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MetricSummary:
    """Summary container holding averaged epoch metrics."""

    loss: float = 0.0
    accuracy: float = 0.0
    perplexity: float = 0.0
    learning_rate: float = 0.0
    grad_norm: float = 0.0
    total_steps: int = 0


class MetricTracker:
    """Accumulates and calculates average training metrics over epochs.

    Time Complexity:
        O(1) update operations.

    Space Complexity:
        O(1) scalar memory overhead.
    """

    def __init__(self) -> None:
        """Initializes MetricTracker."""
        self.reset()

    def reset(self) -> None:
        """Resets all internal metric accumulators."""
        self.total_loss: float = 0.0
        self.total_accuracy: float = 0.0
        self.total_perplexity: float = 0.0
        self.last_lr: float = 0.0
        self.last_grad_norm: float = 0.0
        self.step_count: int = 0

    def update(
        self,
        loss: float,
        accuracy: float = 0.0,
        perplexity: float = 0.0,
        lr: float = 0.0,
        grad_norm: float = 0.0,
    ) -> None:
        """Updates metric accumulators with step results.

        Args:
            loss: Step loss float.
            accuracy: Step accuracy percentage float.
            perplexity: Step perplexity float.
            lr: Step learning rate float.
            grad_norm: Step gradient norm float.
        """
        self.total_loss += loss
        self.total_accuracy += accuracy
        self.total_perplexity += perplexity
        self.last_lr = lr
        if grad_norm > 0.0:
            self.last_grad_norm = grad_norm
        self.step_count += 1

    def get_summary(self) -> MetricSummary:
        """Calculates average metric values over accumulated steps.

        Returns:
            MetricSummary diagnostic report object.
        """
        if self.step_count == 0:
            return MetricSummary()

        avg_loss = round(self.total_loss / self.step_count, 6)
        avg_acc = round(self.total_accuracy / self.step_count, 2)
        avg_ppl = round(self.total_perplexity / self.step_count, 4)

        return MetricSummary(
            loss=avg_loss,
            accuracy=avg_acc,
            perplexity=avg_ppl,
            learning_rate=self.last_lr,
            grad_norm=self.last_grad_norm,
            total_steps=self.step_count,
        )

    def to_dict(self) -> Dict[str, float]:
        """Converts current summary metrics to a dictionary representation.

        Returns:
            Dictionary mapping metric names to floats.
        """
        summary = self.get_summary()
        return {
            "loss": summary.loss,
            "accuracy": summary.accuracy,
            "perplexity": summary.perplexity,
            "lr": summary.learning_rate,
            "grad_norm": summary.grad_norm,
            "steps": float(summary.total_steps),
        }
