"""Training Resume Manager for Aura LLM Architecture.

Restores complete training execution states (model parameters, optimizer moment buffers,
learning rate scheduler steps, AMP scaler, and RNG seeds) from a checkpoint payload.
"""

import logging
import os
from typing import Any, Optional, Tuple
import torch
import torch.nn as nn

from src.utils.checkpoint_loader import CheckpointLoader

logger = logging.getLogger(__name__)


class TrainingResumeManager:
    """Orchestrates bit-exact training resumption from a saved checkpoint."""

    @staticmethod
    def resume_training_state(
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
    ) -> Tuple[int, int]:
        """Restores model, optimizer, scheduler, AMP scaler, and RNG states.

        Args:
            checkpoint_path: Path to checkpoint file (.pt).
            model: PyTorch nn.Module instance.
            optimizer: Optional PyTorch Optimizer or OptimizationManager.
            scheduler: Optional PyTorch _LRScheduler.
            scaler: Optional PyTorch GradScaler.

        Returns:
            Tuple of (epoch, global_step).
        """
        device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
        payload = CheckpointLoader.load_payload(checkpoint_path, device=device)

        # Restore Model Weights
        model_state = payload.get("model_state_dict", payload.get("model", payload))
        model.load_state_dict(model_state)

        # Restore Optimizer State
        if optimizer and "optimizer_state_dict" in payload and payload["optimizer_state_dict"]:
            if hasattr(optimizer, "load_state_dict"):
                optimizer.load_state_dict(payload["optimizer_state_dict"])
            elif hasattr(optimizer, "optimizer") and hasattr(optimizer.optimizer, "load_state_dict"):
                optimizer.optimizer.load_state_dict(payload["optimizer_state_dict"])

        # Restore Scheduler State
        if scheduler and "scheduler_state_dict" in payload and payload["scheduler_state_dict"]:
            if hasattr(scheduler, "load_state_dict"):
                scheduler.load_state_dict(payload["scheduler_state_dict"])
            elif hasattr(optimizer, "scheduler") and hasattr(optimizer.scheduler, "load_state_dict"):
                optimizer.scheduler.load_state_dict(payload["scheduler_state_dict"])

        # Restore Scaler State
        if scaler and "scaler_state_dict" in payload and payload["scaler_state_dict"]:
            if hasattr(scaler, "load_state_dict"):
                scaler.load_state_dict(payload["scaler_state_dict"])

        # Restore RNG States
        if "rng_states" in payload and payload["rng_states"]:
            rng = payload["rng_states"]
            if rng.get("torch_cpu_rng") is not None:
                torch.set_rng_state(rng["torch_cpu_rng"])
            if rng.get("torch_cuda_rng") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["torch_cuda_rng"])

        epoch = payload.get("epoch", 0)
        global_step = payload.get("global_step", 0)

        logger.info("TrainingResumeManager: Successfully restored state from %s (epoch %d, step %d)", checkpoint_path, epoch, global_step)
        return epoch, global_step
