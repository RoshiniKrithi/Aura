"""Optimization Statistics Extractor for Aura LLM Architecture.

Computes parameter group breakdown, current learning rates, total gradient norm metrics,
and step counters across optimization updates.
"""

from dataclasses import dataclass
import logging
from typing import Dict, Tuple
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class OptimizationStats:
    """Summary container holding optimization metrics."""

    learning_rate: float
    max_grad_norm: float
    decay_params_count: int
    no_decay_params_count: int
    global_step: int
    micro_step: int


class OptimizationStatistics:
    """Computes quantitative statistical metrics over OptimizationManager components."""

    @staticmethod
    def compute_stats(
        model: nn.Module,
        current_lr: float = 0.0,
        max_grad_norm: float = 1.0,
        global_step: int = 0,
        micro_step: int = 0,
    ) -> OptimizationStats:
        """Calculates model parameter group counts and optimization status.

        Args:
            model: PyTorch nn.Module instance.
            current_lr: Active learning rate float.
            max_grad_norm: Maximum gradient norm clipping threshold.
            global_step: Current global training step.
            micro_step: Current micro-batch step.

        Returns:
            OptimizationStats summary object.
        """
        decay_count = 0
        no_decay_count = 0

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if param.ndim < 2 or "bias" in name or "ln_" in name or "norm" in name:
                no_decay_count += param.numel()
            else:
                decay_count += param.numel()

        return OptimizationStats(
            learning_rate=current_lr,
            max_grad_norm=max_grad_norm,
            decay_params_count=decay_count,
            no_decay_params_count=no_decay_count,
            global_step=global_step,
            micro_step=micro_step,
        )
