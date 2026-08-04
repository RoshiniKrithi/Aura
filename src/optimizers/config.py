"""Optimizer Configuration Schema for Aura LLM Architecture.

Provides parameters for optimizer selection (AdamW, SGD, Adam), learning rates,
weight decay, momentum beta1/beta2 parameters, epsilon constants, and unified OptimizationConfig.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.schedulers.config import SchedulerConfig


@dataclass(frozen=True)
class OptimizerConfig:
    """Hyperparameter configuration container for Optimizer algorithms.

    Attributes:
        name: Optimizer algorithm name ("adamw", "sgd", "adam").
        lr: Base peak learning rate (default: 3e-4).
        weight_decay: Decoupled weight decay coefficient (default: 0.1).
        beta1: Adam/AdamW first moment momentum coefficient (default: 0.9).
        beta2: Adam/AdamW second moment momentum coefficient (default: 0.95).
        eps: Epsilon constant for numerical stability (default: 1e-8).
        filter_weight_decay: If True, excludes 1D biases and norm parameters from weight decay (default: True).
    """

    name: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    filter_weight_decay: bool = True


@dataclass(frozen=True)
class OptimizationConfig:
    """Unified configuration container for complete Optimization system.

    Attributes:
        optimizer: OptimizerConfig hyperparameter object.
        scheduler: SchedulerConfig hyperparameter object.
        max_grad_norm: Maximum L2 gradient norm threshold for clipping (default: 1.0).
        gradient_accumulation_steps: Micro-batch steps before optimizer update (default: 1).
    """

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
