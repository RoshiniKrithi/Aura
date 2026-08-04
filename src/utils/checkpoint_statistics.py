"""Checkpoint Statistics Summary for Aura LLM Architecture.

Computes summary metrics over saved checkpoint directory (file counts, disk usage bytes, best metric value).
"""

from dataclasses import dataclass
import glob
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CheckpointStats:
    """Summary container for checkpoint storage statistics."""

    total_checkpoints: int
    total_disk_bytes: int
    latest_step: int
    has_best_model: bool


class CheckpointStatistics:
    """Calculates quantitative storage and versioning statistics."""

    @staticmethod
    def compute_stats(checkpoint_dir: str = "checkpoints") -> CheckpointStats:
        """Computes summary statistics for a checkpoint directory.

        Args:
            checkpoint_dir: Path to directory containing checkpoints.

        Returns:
            CheckpointStats summary object.
        """
        if not os.path.exists(checkpoint_dir):
            return CheckpointStats(
                total_checkpoints=0,
                total_disk_bytes=0,
                latest_step=0,
                has_best_model=False,
            )

        pattern = os.path.join(checkpoint_dir, "*.pt")
        files = glob.glob(pattern)

        has_best = any(os.path.basename(f) == "best_model.pt" for f in files)
        step_files = [f for f in files if os.path.basename(f) != "best_model.pt"]

        total_bytes = sum(os.path.getsize(f) for f in files)
        latest_step = 0

        for f in step_files:
            try:
                parts = os.path.basename(f).replace(".pt", "").split("_")
                if "step" in parts:
                    idx = parts.index("step")
                    if idx + 1 < len(parts):
                        latest_step = max(latest_step, int(parts[idx + 1]))
            except ValueError:
                pass

        return CheckpointStats(
            total_checkpoints=len(step_files),
            total_disk_bytes=total_bytes,
            latest_step=latest_step,
            has_best_model=has_best,
        )
