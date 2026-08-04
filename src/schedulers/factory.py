"""Scheduler Factory API for Aura LLM Architecture.

Provides factory builder methods for constructing learning rate schedulers
from SchedulerConfig, AppConfig, or string algorithm names.
"""

import logging
from typing import Any, Optional, Union
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ConstantLR, StepLR, _LRScheduler

from src.schedulers.config import SchedulerConfig
from src.schedulers.cosine_warmup import CosineAnnealingWithWarmupLR

logger = logging.getLogger(__name__)


class SchedulerFactory:
    """Central factory for constructing and initializing learning rate schedulers."""

    @classmethod
    def create_scheduler(
        cls,
        optimizer: Optimizer,
        config: Optional[Union[SchedulerConfig, Any]] = None,
        name: Optional[str] = None,
        warmup_steps: Optional[int] = None,
        max_steps: Optional[int] = None,
        min_lr: Optional[float] = None,
    ) -> _LRScheduler:
        """Instantiates and configures a PyTorch learning rate scheduler.

        Args:
            optimizer: Target PyTorch Optimizer instance.
            config: Optional SchedulerConfig or AppConfig object.
            name: Optional scheduler name override ("cosine_warmup", "step", "constant").
            warmup_steps: Optional warmup steps override.
            max_steps: Optional max decay steps override.
            min_lr: Optional min learning rate floor override.

        Returns:
            Instantiated PyTorch _LRScheduler instance.
        """
        from src.utils.config import AppConfig

        if isinstance(config, AppConfig):
            cfg = SchedulerConfig(
                name=name or config.scheduler.type if hasattr(config.scheduler, "type") else "cosine_warmup",
                warmup_steps=warmup_steps if warmup_steps is not None else getattr(config.scheduler, "warmup_steps", 2000),
                max_steps=max_steps if max_steps is not None else getattr(config.scheduler, "max_steps", 100000),
                min_lr=min_lr if min_lr is not None else getattr(config.scheduler, "min_lr", 3e-5),
            )
        elif isinstance(config, SchedulerConfig):
            cfg = config
        else:
            cfg = SchedulerConfig(
                name=name or "cosine_warmup",
                warmup_steps=warmup_steps if warmup_steps is not None else 2000,
                max_steps=max_steps if max_steps is not None else 100000,
                min_lr=min_lr if min_lr is not None else 3e-5,
            )

        sched_type = cfg.name.lower().strip()

        if "cosine" in sched_type or "warmup" in sched_type:
            scheduler = CosineAnnealingWithWarmupLR(
                optimizer=optimizer,
                warmup_steps=cfg.warmup_steps,
                max_steps=cfg.max_steps,
                min_lr=cfg.min_lr,
            )
        elif "step" in sched_type:
            scheduler = StepLR(optimizer=optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
        elif "constant" in sched_type:
            scheduler = ConstantLR(optimizer=optimizer, factor=1.0)
        else:
            scheduler = CosineAnnealingWithWarmupLR(
                optimizer=optimizer,
                warmup_steps=cfg.warmup_steps,
                max_steps=cfg.max_steps,
                min_lr=cfg.min_lr,
            )

        logger.info("Instantiated LR Scheduler '%s' bound to optimizer", sched_type)
        return scheduler
