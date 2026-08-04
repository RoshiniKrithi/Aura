"""Training Statistics Summary for Aura LLM Architecture.

Computes quantitative summary statistics over training metrics, total duration,
and step counts across complete training runs.
"""

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrainingStats:
    """Diagnostic report object holding run execution statistics."""

    total_epochs: int
    total_duration_seconds: float
    final_train_loss: float
    final_train_accuracy: float
    final_train_perplexity: float
    final_val_loss: Optional[float] = None
    final_val_accuracy: Optional[float] = None


class TrainingStatistics:
    """Calculates statistical summary diagnostics from training run history."""

    @staticmethod
    def compute_run_stats(history: List[Dict[str, Any]], total_duration: float = 0.0) -> TrainingStats:
        """Computes summary statistics from a training history list.

        Args:
            history: History records list produced by TrainingEngine.fit().
            total_duration: Total training run duration seconds.

        Returns:
            TrainingStats report object.
        """
        if not history:
            return TrainingStats(
                total_epochs=0,
                total_duration_seconds=total_duration,
                final_train_loss=0.0,
                final_train_accuracy=0.0,
                final_train_perplexity=0.0,
            )

        last_record = history[-1]
        train_m = last_record.get("train", {})
        val_m = last_record.get("val", {})

        return TrainingStats(
            total_epochs=len(history),
            total_duration_seconds=total_duration,
            final_train_loss=train_m.get("loss", 0.0),
            final_train_accuracy=train_m.get("accuracy", 0.0),
            final_train_perplexity=train_m.get("perplexity", 0.0),
            final_val_loss=val_m.get("loss") if val_m else None,
            final_val_accuracy=val_m.get("accuracy") if val_m else None,
        )
