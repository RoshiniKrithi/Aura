"""Training Logger for Aura LLM Architecture.

Provides structured logging methods for epoch metrics, step progress, and validation results.
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class TrainingLogger:
    """Handles structured logging output for training events and metric summaries."""

    @staticmethod
    def log_epoch_start(epoch: int, total_epochs: int) -> None:
        """Logs start of an epoch."""
        logger.info("=== Starting Epoch [%d/%d] ===", epoch, total_epochs)

    @staticmethod
    def log_epoch_end(epoch: int, total_epochs: int, train_metrics: Dict[str, float], val_metrics: Optional[Dict[str, float]] = None) -> None:
        """Logs summary metrics at end of an epoch."""
        loss_str = f"Loss: {train_metrics.get('loss', 0.0):.4f}"
        acc_str = f"Acc: {train_metrics.get('accuracy', 0.0):.2f}%"
        ppl_str = f"PPL: {train_metrics.get('perplexity', 0.0):.2f}"
        lr_str = f"LR: {train_metrics.get('lr', 0.0):.6f}"

        msg = f"Epoch [{epoch}/{total_epochs}] Train -> {loss_str} | {acc_str} | {ppl_str} | {lr_str}"

        if val_metrics:
            val_loss = f"Val Loss: {val_metrics.get('loss', 0.0):.4f}"
            val_acc = f"Val Acc: {val_metrics.get('accuracy', 0.0):.2f}%"
            msg += f" || {val_loss} | {val_acc}"

        logger.info(msg)

    @staticmethod
    def log_checkpoint_saved(path: str) -> None:
        """Logs checkpoint saving event."""
        logger.info("Saved model checkpoint file to: %s", path)
