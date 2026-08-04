"""Cosine Annealing with Warmup Learning Rate Scheduler for Aura LLM Architecture.

Warms up learning rate linearly from 0.0 to base_lr over warmup_steps,
then decays learning rate according to a cosine schedule down to min_lr over max_steps.
"""

import math
import logging
from typing import List, Union
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger(__name__)


class CosineAnnealingWithWarmupLR(_LRScheduler):
    """Production-grade Cosine Annealing with Warmup Learning Rate Scheduler.

    Formula:
        - If step < warmup_steps:
            lr_t = base_lr * (step / warmup_steps)
        - If warmup_steps <= step <= max_steps:
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            lr_t = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * progress))
        - If step > max_steps:
            lr_t = min_lr
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int = 2000,
        max_steps: int = 100000,
        min_lr: float = 3e-5,
        last_epoch: int = -1,
    ) -> None:
        """Initializes CosineAnnealingWithWarmupLR.

        Args:
            optimizer: Wrapped PyTorch Optimizer instance.
            warmup_steps: Number of linear warmup steps.
            max_steps: Total training steps for cosine decay completion.
            min_lr: Floor minimum learning rate.
            last_epoch: Index of last step (-1 for initialization).
        """
        self.warmup_steps = max(0, warmup_steps)
        self.max_steps = max(1, max_steps)
        self.min_lr = min_lr

        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> List[float]:
        """Calculates learning rates for all parameter groups at current step."""
        step = self.last_epoch

        if step < self.warmup_steps:
            if self.warmup_steps == 0:
                return [base_lr for base_lr in self.base_lrs]
            alpha = float(step) / float(self.warmup_steps)
            return [base_lr * alpha for base_lr in self.base_lrs]

        if step > self.max_steps:
            return [self.min_lr for _ in self.base_lrs]

        decay_steps = self.max_steps - self.warmup_steps
        if decay_steps <= 0:
            return [self.min_lr for _ in self.base_lrs]

        progress = float(step - self.warmup_steps) / float(decay_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

        return [
            self.min_lr + (base_lr - self.min_lr) * cosine_decay
            for base_lr in self.base_lrs
        ]
