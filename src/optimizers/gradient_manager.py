"""Gradient Manager for Aura LLM Architecture.

Provides gradient norm clipping (C=1.0), zeroing parameter gradients (set_to_none=True),
and gradient accumulation step trigger tracking.
"""

import logging
from typing import Iterable, Optional
import torch
import torch.nn as nn
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


class GradientManager:
    """Manages gradient norms, clipping, zeroing, and accumulation logic.

    Time Complexity:
        O(P) norm scan over parameter gradients.

    Space Complexity:
        O(1) scalar memory overhead.
    """

    def __init__(self, max_grad_norm: float = 1.0, accumulation_steps: int = 1) -> None:
        """Initializes GradientManager.

        Args:
            max_grad_norm: Maximum threshold for L2 gradient norm clipping (default: 1.0).
            accumulation_steps: Number of micro-batch steps before optimizer update (default: 1).
        """
        self.max_grad_norm = max_grad_norm
        self.accumulation_steps = max(1, accumulation_steps)

    def should_step(self, micro_step: int) -> bool:
        """Determines if optimizer update step should be triggered at current micro_step.

        Args:
            micro_step: Current micro-batch iteration step integer (1-indexed or 0-indexed).

        Returns:
            True if (micro_step + 1) % accumulation_steps == 0.
        """
        return (micro_step + 1) % self.accumulation_steps == 0

    def clip_grad_norm(self, parameters: Iterable[torch.Tensor]) -> float:
        """Clips L2 gradient norm of parameters to max_grad_norm threshold.

        Args:
            parameters: Iterable of parameter Tensors with computed gradients.

        Returns:
            Total computed L2 gradient norm float before clipping.
        """
        if self.max_grad_norm <= 0.0:
            return 0.0

        params_with_grad = [p for p in parameters if p.grad is not None]
        if not params_with_grad:
            return 0.0

        grad_norm = torch.nn.utils.clip_grad_norm_(params_with_grad, self.max_grad_norm)
        return float(grad_norm.item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)

    @staticmethod
    def zero_grad(optimizer: Optimizer, set_to_none: bool = True) -> None:
        """Zeroes parameter gradients in optimizer using memory-efficient set_to_none=True.

        Args:
            optimizer: Target PyTorch Optimizer instance.
            set_to_none: If True, sets parameter .grad attributes to None to save memory.
        """
        optimizer.zero_grad(set_to_none=set_to_none)
