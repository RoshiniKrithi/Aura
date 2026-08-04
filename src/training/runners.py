"""Training & Validation Runners for Aura LLM Architecture.

Provides BatchRunner, EpochRunner, and ValidationRunner modules implementing Single Responsibility Principle (SRP)
for batch execution, epoch loops, and validation evaluation.
"""

import logging
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.losses.cross_entropy import CrossEntropyLoss
from src.optimizers.manager import OptimizationManager
from src.training.config import TrainingEngineConfig
from src.training.exceptions import EngineValidationError
from src.training.metrics import MetricTracker

logger = logging.getLogger(__name__)


class BatchRunner:
    """Executes single micro-batch forward, loss computation, backward, and optimizer step.

    Design Decisions:
        - Decoupled from epoch state.
        - Supports Automatic Mixed Precision (AMP) via torch.amp.autocast.
        - Micro-batch gradient scaling for accumulation.
    """

    @staticmethod
    def run_batch(
        model: nn.Module,
        batch: Any,
        loss_module: CrossEntropyLoss,
        optimization_manager: OptimizationManager,
        device: torch.device,
        micro_step: int,
        config: TrainingEngineConfig,
        scaler: Optional[Any] = None,
    ) -> Tuple[float, Dict[str, float], bool, float, float]:
        """Executes a single micro-batch step.

        Args:
            model: PyTorch nn.Module instance.
            batch: Batch tuple, list, or dict representation.
            loss_module: CrossEntropyLoss module instance.
            optimization_manager: OptimizationManager instance.
            device: Target torch.device.
            micro_step: Micro-batch step integer.
            config: TrainingEngineConfig object.
            scaler: Optional GradScaler for AMP.

        Returns:
            Tuple of (loss_item, loss_metrics_dict, did_step, current_lr, grad_norm).
        """
        input_ids, targets = BatchRunner._parse_batch(batch)
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        accum_steps = optimization_manager.gradient_manager.accumulation_steps
        device_type = device.type if hasattr(device, "type") else "cpu"
        if device_type not in ["cuda", "cpu"]:
            device_type = "cpu"

        amp_dtype = torch.bfloat16 if config.amp_dtype == "bfloat16" else torch.float16

        with torch.amp.autocast(device_type=device_type, enabled=config.amp_enabled, dtype=amp_dtype):
            logits = model(input_ids)
            loss, loss_metrics = loss_module(logits, targets)
            scaled_loss = loss / accum_steps

        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        did_step, current_lr, grad_norm = optimization_manager.step(
            micro_step=micro_step, scaler=scaler
        )

        return loss.item(), loss_metrics, did_step, current_lr, grad_norm

    @staticmethod
    def _parse_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extracts input_ids and targets from batch structure."""
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            return batch[0], batch[1]
        elif isinstance(batch, dict):
            if "input_ids" in batch and "targets" in batch:
                return batch["input_ids"], batch["targets"]
            elif "input_ids" in batch and "labels" in batch:
                return batch["input_ids"], batch["labels"]
            elif "input_ids" in batch:
                return batch["input_ids"], batch["input_ids"]
        elif isinstance(batch, torch.Tensor):
            return batch, batch

        raise EngineValidationError(f"Unsupported batch structure type: {type(batch).__name__}")


class EpochRunner:
    """Orchestrates training execution over a full DataLoader epoch."""

    @staticmethod
    def run_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        loss_module: CrossEntropyLoss,
        optimization_manager: OptimizationManager,
        device: torch.device,
        config: TrainingEngineConfig,
        tracker: MetricTracker,
        scaler: Optional[Any] = None,
    ) -> Dict[str, float]:
        """Runs a complete training epoch.

        Args:
            model: Target PyTorch nn.Module.
            dataloader: Training DataLoader instance.
            loss_module: CrossEntropyLoss module instance.
            optimization_manager: OptimizationManager instance.
            device: Target torch.device.
            config: TrainingEngineConfig object.
            tracker: MetricTracker accumulator object.
            scaler: Optional GradScaler for AMP.

        Returns:
            Dictionary of epoch summary metrics.
        """
        model.train()
        tracker.reset()

        for micro_step, batch in enumerate(dataloader):
            loss_val, metrics, did_step, current_lr, grad_norm = BatchRunner.run_batch(
                model=model,
                batch=batch,
                loss_module=loss_module,
                optimization_manager=optimization_manager,
                device=device,
                micro_step=micro_step,
                config=config,
                scaler=scaler,
            )

            tracker.update(
                loss=loss_val,
                accuracy=metrics.get("accuracy", 0.0),
                perplexity=metrics.get("perplexity", 0.0),
                lr=current_lr,
                grad_norm=grad_norm,
            )

            if config.max_steps and micro_step >= config.max_steps:
                break

        return tracker.to_dict()


class ValidationRunner:
    """Executes validation evaluation loop in model.eval() mode."""

    @staticmethod
    def run_validation(
        model: nn.Module,
        dataloader: DataLoader,
        loss_module: CrossEntropyLoss,
        device: torch.device,
        tracker: MetricTracker,
    ) -> Dict[str, float]:
        """Runs validation evaluation over dataloader.

        Args:
            model: Target PyTorch nn.Module.
            dataloader: Validation DataLoader instance.
            loss_module: CrossEntropyLoss module.
            device: Target torch.device.
            tracker: MetricTracker accumulator object.

        Returns:
            Dictionary of averaged validation metrics.
        """
        model.eval()
        tracker.reset()

        with torch.no_grad():
            for batch in dataloader:
                input_ids, targets = BatchRunner._parse_batch(batch)
                input_ids = input_ids.to(device)
                targets = targets.to(device)

                logits = model(input_ids)
                loss, loss_metrics = loss_module(logits, targets)

                tracker.update(
                    loss=loss.item(),
                    accuracy=loss_metrics.get("accuracy", 0.0),
                    perplexity=loss_metrics.get("perplexity", 0.0),
                )

        return tracker.to_dict()
