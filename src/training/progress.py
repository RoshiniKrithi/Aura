"""Progress Tracker for Aura LLM Architecture.

Tracks training micro-step counts, progress percentages, and timing metrics.
"""

import time
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Tracks training step counts, elapsed time, and progress metrics."""

    def __init__(self, total_epochs: int = 10, total_steps_per_epoch: int = 100) -> None:
        """Initializes ProgressTracker.

        Args:
            total_epochs: Total epoch count integer.
            total_steps_per_epoch: Micro-steps per epoch integer.
        """
        self.total_epochs = max(1, total_epochs)
        self.total_steps_per_epoch = max(1, total_steps_per_epoch)
        self.total_steps = self.total_epochs * self.total_steps_per_epoch

        self.current_epoch = 0
        self.current_step = 0
        self.start_time = time.time()

    def update(self, epoch: int, step: int) -> None:
        """Updates current epoch and step counters.

        Args:
            epoch: Current 1-indexed epoch integer.
            step: Current 0-indexed micro-step integer.
        """
        self.current_epoch = epoch
        self.current_step = (epoch - 1) * self.total_steps_per_epoch + step + 1

    def get_progress_pct(self) -> float:
        """Calculates total progress percentage float (0.0 to 100.0)."""
        if self.total_steps == 0:
            return 0.0
        return round(min(100.0, (self.current_step / self.total_steps) * 100.0), 2)

    def get_elapsed_seconds(self) -> float:
        """Calculates elapsed seconds float since tracking started."""
        return round(time.time() - self.start_time, 2)

    def to_dict(self) -> Dict[str, float]:
        """Returns dictionary representation of progress state."""
        return {
            "current_epoch": float(self.current_epoch),
            "total_epochs": float(self.total_epochs),
            "current_step": float(self.current_step),
            "total_steps": float(self.total_steps),
            "progress_pct": self.get_progress_pct(),
            "elapsed_seconds": self.get_elapsed_seconds(),
        }
