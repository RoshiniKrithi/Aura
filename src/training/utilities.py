"""Training Engine Helper Utilities for Aura LLM Architecture.

Provides helper functions for setting seed reproducibility, formatting training stats,
and device placement checks.
"""

import logging
import os
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EngineUtilities:
    """Helper utility functions for TrainingEngine."""

    @staticmethod
    def set_seed(seed: int = 42) -> None:
        """Sets global random seed across Python, NumPy, and PyTorch.

        Args:
            seed: Integer random seed value.
        """
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info("Set random seed to %d for reproducibility.", seed)

    @staticmethod
    def format_epoch_summary(epoch: int, total_epochs: int, metrics: dict) -> str:
        """Formats epoch metrics into a readable log string.

        Args:
            epoch: Current epoch index (1-indexed).
            total_epochs: Total epoch count.
            metrics: Dictionary containing epoch metrics.

        Returns:
            Formatted log string.
        """
        loss_str = f"loss: {metrics.get('loss', 0.0):.4f}"
        acc_str = f"acc: {metrics.get('accuracy', 0.0):.2f}%"
        ppl_str = f"ppl: {metrics.get('perplexity', 0.0):.2f}"
        lr_str = f"lr: {metrics.get('lr', 0.0):.6f}"

        return f"Epoch [{epoch}/{total_epochs}] - {loss_str} | {acc_str} | {ppl_str} | {lr_str}"
