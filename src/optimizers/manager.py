"""Unified Optimization Manager for Aura LLM Architecture.

Orchestrates forward-backward step accumulation, gradient clipping, optimizer steps,
Automatic Mixed Precision (AMP) scaler updates, and learning rate scheduler steps in a single production API.
"""

import logging
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from src.optimizers.config import OptimizationConfig
from src.optimizers.factory import OptimizerFactory
from src.optimizers.gradient_manager import GradientManager
from src.schedulers.factory import SchedulerFactory

logger = logging.getLogger(__name__)


class OptimizationManager:
    """Unified Orchestrator for Optimizers, Schedulers, Gradient Clipping & Accumulation.

    Design Decisions:
        - Abstracts micro-batch gradient accumulation step triggers.
        - Applies L2 gradient norm clipping (C=1.0) before optimizer steps.
        - Supports Automatic Mixed Precision (AMP) GradScaler hooks.
        - Steps learning rate scheduler and zeroes parameter gradients (set_to_none=True).
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[OptimizationConfig] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        scaler: Optional[Any] = None,
    ) -> None:
        """Initializes OptimizationManager binding model to optimizer, scheduler, and AMP scaler.

        Args:
            model: Target PyTorch nn.Module model.
            config: Optional OptimizationConfig object.
            optimizer: Optional pre-constructed PyTorch Optimizer instance.
            scheduler: Optional pre-constructed PyTorch _LRScheduler instance.
            scaler: Optional PyTorch GradScaler instance for Automatic Mixed Precision (AMP).
        """
        cfg = config or OptimizationConfig()
        self.config = cfg
        self.model = model
        self.scaler = scaler

        # 1. Instantiate Optimizer via Factory if not provided
        self.optimizer = optimizer or OptimizerFactory.create_optimizer(
            model=model, config=cfg.optimizer
        )

        # 2. Instantiate Scheduler via Factory if not provided
        self.scheduler = scheduler or SchedulerFactory.create_scheduler(
            optimizer=self.optimizer, config=cfg.scheduler
        )

        # 3. Instantiate Gradient Manager for clipping & accumulation
        self.gradient_manager = GradientManager(
            max_grad_norm=cfg.max_grad_norm,
            accumulation_steps=cfg.gradient_accumulation_steps,
        )

        self.global_step = 0
        self.micro_step = 0

        logger.info(
            "Instantiated OptimizationManager: max_grad_norm=%.2f, accumulation_steps=%d, amp_scaler=%s",
            cfg.max_grad_norm,
            cfg.gradient_accumulation_steps,
            self.scaler is not None,
        )

    def step(
        self, micro_step: Optional[int] = None, scaler: Optional[Any] = None
    ) -> Tuple[bool, float, float]:
        """Executes accumulation tracking, gradient clipping, optimizer step, and scheduler step.

        Args:
            micro_step: Optional iteration micro-step index integer. If None, uses internal counter.
            scaler: Optional GradScaler override for mixed precision steps.

        Returns:
            Tuple of (did_optimizer_step, current_lr, grad_norm).
        """
        step_idx = micro_step if micro_step is not None else self.micro_step
        self.micro_step = step_idx + 1

        should_update = self.gradient_manager.should_step(step_idx)
        active_scaler = scaler if scaler is not None else self.scaler

        current_lr = self.get_lr()
        grad_norm = 0.0

        if should_update:
            if active_scaler is not None:
                # 1. Unscale gradients before clipping
                active_scaler.unscale_(self.optimizer)
                # 2. Clip gradient norms
                grad_norm = self.gradient_manager.clip_grad_norm(self.model.parameters())
                # 3. Step optimizer via GradScaler
                active_scaler.step(self.optimizer)
                active_scaler.update()
            else:
                # 1. Clip gradient norms
                grad_norm = self.gradient_manager.clip_grad_norm(self.model.parameters())
                # 2. Step optimizer
                self.optimizer.step()

            # 3. Execute learning rate scheduler step
            self.scheduler.step()

            # 4. Zero parameter gradients using set_to_none=True
            self.gradient_manager.zero_grad(self.optimizer, set_to_none=True)

            self.global_step += 1
            current_lr = self.get_lr()

            return True, current_lr, grad_norm

        return False, current_lr, 0.0

    def get_lr(self) -> float:
        """Retrieves current learning rate float from optimizer / scheduler."""
        if hasattr(self.scheduler, "get_last_lr"):
            lrs = self.scheduler.get_last_lr()
            if lrs:
                return float(lrs[0])
        return float(self.optimizer.param_groups[0]["lr"])

    def zero_grad(self) -> None:
        """Zeroes parameter gradients in optimizer."""
        self.gradient_manager.zero_grad(self.optimizer, set_to_none=True)

    def state_dict(self) -> Dict[str, Any]:
        """Serializes optimizer, scheduler, and AMP scaler state dicts for checkpointing.

        Returns:
            Dictionary containing optimizer_state_dict, scheduler_state_dict, scaler_state_dict, global_step, and micro_step.
        """
        state = {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "micro_step": self.micro_step,
        }
        if self.scaler is not None and hasattr(self.scaler, "state_dict"):
            state["scaler_state_dict"] = self.scaler.state_dict()
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restores optimizer, scheduler, and AMP scaler states from checkpoint.

        Args:
            state_dict: Serialized dictionary from state_dict().
        """
        if "optimizer_state_dict" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        elif "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer"])

        if "scheduler_state_dict" in state_dict:
            self.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        elif "scheduler" in state_dict:
            self.scheduler.load_state_dict(state_dict["scheduler"])

        if self.scaler is not None and "scaler_state_dict" in state_dict and hasattr(self.scaler, "load_state_dict"):
            self.scaler.load_state_dict(state_dict["scaler_state_dict"])

        self.global_step = state_dict.get("global_step", 0)
        self.micro_step = state_dict.get("micro_step", 0)

        logger.info("Restored OptimizationManager checkpoint states: global_step=%d", self.global_step)
