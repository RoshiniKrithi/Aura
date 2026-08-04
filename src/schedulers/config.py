"""Scheduler Configuration Schema for Aura LLM Architecture.

Provides parameters for learning rate scheduler selection, warmup steps,
max decay steps, minimum learning rate, step sizes, and decay factors.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class SchedulerConfig:
    """Hyperparameter configuration container for learning rate schedulers.

    Attributes:
        name: Scheduler algorithm name ("cosine_warmup", "linear_warmup", "step", "constant").
        warmup_steps: Number of linear warmup steps at training start (default: 2000).
        max_steps: Maximum total training steps for cosine decay (default: 100000).
        min_lr: Minimum learning rate floor after decay (default: 3e-5).
        step_size: Step size period for StepLR decay (default: 10000).
        gamma: Decay multiplier factor for StepLR (default: 0.5).
    """

    name: str = "cosine_warmup"
    warmup_steps: int = 2000
    max_steps: int = 100000
    min_lr: float = 3e-5
    step_size: int = 10000
    gamma: float = 0.5
