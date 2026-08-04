"""Optimization Utilities for Aura LLM Architecture.

Provides WeightDecayUtilities for parameter decay group separation
and CheckpointUtilities for optimizer/scheduler state dict serialization and restoration.
"""

import logging
from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger(__name__)


class WeightDecayUtilities:
    """Helper utilities for parameter weight decay filtering."""

    @staticmethod
    def filter_weight_decay_params(
        model: nn.Module, weight_decay: float = 0.1
    ) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        """Separates model parameters into 2D decay weights and 1D non-decay parameters.

        Args:
            model: Target PyTorch nn.Module.
            weight_decay: Weight decay coefficient float.

        Returns:
            Tuple of (decay_params_list, no_decay_params_list).
        """
        decay_params = []
        no_decay_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if param.ndim < 2 or "bias" in name or "ln_" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        return decay_params, no_decay_params


class CheckpointUtilities:
    """Helper utilities for optimizer and scheduler state dict serialization."""

    @staticmethod
    def create_optimization_checkpoint(
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        global_step: int = 0,
        micro_step: int = 0,
    ) -> Dict[str, Any]:
        """Creates a state dict checkpoint dictionary from optimizer and scheduler.

        Args:
            optimizer: Active PyTorch Optimizer.
            scheduler: Active PyTorch _LRScheduler.
            global_step: Current global step integer.
            micro_step: Current micro step integer.

        Returns:
            Checkpoint state dictionary.
        """
        return {
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "global_step": global_step,
            "micro_step": micro_step,
        }

    @staticmethod
    def restore_optimization_checkpoint(
        checkpoint: Dict[str, Any],
        optimizer: Optimizer,
        scheduler: _LRScheduler,
    ) -> Tuple[int, int]:
        """Restores optimizer and scheduler states from a checkpoint dictionary.

        Args:
            checkpoint: Serialized state dictionary.
            optimizer: Target PyTorch Optimizer instance.
            scheduler: Target PyTorch _LRScheduler instance.

        Returns:
            Tuple of (global_step, micro_step).
        """
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        elif "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])

        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        elif "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])

        global_step = checkpoint.get("global_step", 0)
        micro_step = checkpoint.get("micro_step", 0)

        logger.info("Restored optimizer and scheduler checkpoint at global step %d", global_step)
        return global_step, micro_step
