"""Production-Grade Training Engine Implementation for Aura LLM Architecture.

Orchestrates end-to-end training and evaluation loops, Automatic Mixed Precision (AMP),
gradient accumulation micro-batching, validation flow, metric tracking, and checkpointing hooks.
"""

import os
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.losses.cross_entropy import CrossEntropyLoss
from src.losses.factory import LossFactory
from src.optimizers.manager import OptimizationManager
from src.training.config import TrainingEngineConfig
from src.training.exceptions import EngineValidationError
from src.training.logger import TrainingLogger
from src.training.metrics import MetricTracker
from src.training.progress import ProgressTracker
from src.training.runners import EpochRunner, ValidationRunner
from src.training.statistics import TrainingStatistics, TrainingStats
from src.training.utilities import EngineUtilities
from src.training.validator import EngineValidator

logger = logging.getLogger(__name__)


class TrainingEngine:
    """Production-grade Training Engine orchestrator.

    Design Decisions:
        - Coordinates DataLoader ingestion, model forward pass, loss computation, and AdamW optimization steps.
        - Supports Automatic Mixed Precision (AMP) under autocast context.
        - Micro-batch gradient accumulation (loss / accumulation_steps).
        - Separates training mode (model.train()) from validation mode (model.eval(), no_grad()).
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        config: Optional[TrainingEngineConfig] = None,
        optimization_manager: Optional[OptimizationManager] = None,
        loss_module: Optional[CrossEntropyLoss] = None,
    ) -> None:
        """Initializes TrainingEngine binding model, data, loss, and optimizer.

        Args:
            model: PyTorch nn.Module model instance.
            train_dataloader: Training DataLoader instance.
            val_dataloader: Optional Validation DataLoader instance.
            config: Optional TrainingEngineConfig hyperparameter object.
            optimization_manager: Optional pre-constructed OptimizationManager instance.
            loss_module: Optional pre-constructed CrossEntropyLoss instance.

        Raises:
            EngineValidationError: If setup validation checks fail.
        """
        val_res = EngineValidator.validate_setup(model, train_dataloader, val_dataloader)
        if not val_res.is_valid:
            raise EngineValidationError(f"Engine initialization validation failed: {val_res.errors}")

        cfg = config or TrainingEngineConfig()
        self.config = cfg
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Set seed for reproducibility
        EngineUtilities.set_seed(cfg.seed)

        # Target Device Setup
        self.device = self._resolve_device(cfg.device)
        try:
            self.model = self.model.to(self.device)
        except Exception as e:
            logger.warning("Moving model to device '%s' failed: %s", self.device, str(e))

        # Loss Module Setup
        self.loss_module = loss_module or LossFactory.create_loss()

        # Optimization Manager Setup
        self.optimization_manager = optimization_manager or OptimizationManager(model=self.model)

        # Metric Trackers
        self.train_tracker = MetricTracker()
        self.val_tracker = MetricTracker()

        # Progress Tracker
        steps_per_epoch = len(train_dataloader) if hasattr(train_dataloader, "__len__") else 100
        self.progress_tracker = ProgressTracker(total_epochs=cfg.epochs, total_steps_per_epoch=steps_per_epoch)

        # AMP GradScaler setup if enabled
        self.scaler = None
        if cfg.amp_enabled and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)
            self.optimization_manager.scaler = self.scaler

        self.current_epoch = 0

        logger.info(
            "Instantiated TrainingEngine: epochs=%d, device=%s, amp_enabled=%s, eval_interval=%d",
            cfg.epochs,
            self.device,
            cfg.amp_enabled,
            cfg.eval_interval,
        )

    def _resolve_device(self, target_device: str) -> torch.device:
        """Resolves device string to PyTorch torch.device object."""
        if target_device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(target_device)

    def fit(self) -> Dict[str, Any]:
        """Executes full multi-epoch training and validation pipeline.

        Returns:
            Dictionary containing epoch metrics history, validation metrics, and total duration.
        """
        logger.info("Starting training run for %d epochs...", self.config.epochs)
        start_time = time.time()
        history: List[Dict[str, Any]] = []

        for epoch in range(1, self.config.epochs + 1):
            self.current_epoch = epoch
            TrainingLogger.log_epoch_start(epoch, self.config.epochs)

            # 1. Execute Training Epoch Loop via EpochRunner
            train_metrics = self.train_epoch(epoch)

            # 2. Execute Epoch Validation if val_dataloader is provided
            val_metrics: Optional[Dict[str, float]] = None
            if self.val_dataloader is not None and (epoch % self.config.eval_interval == 0):
                val_metrics = self.validate()

            TrainingLogger.log_epoch_end(epoch, self.config.epochs, train_metrics, val_metrics)

            epoch_record = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
            history.append(epoch_record)

            # 3. Save Checkpoint at specified intervals
            if epoch % self.config.checkpoint_interval == 0:
                ckpt_dir = self.config.checkpoint_dir
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch}.pt")
                self.save_checkpoint(ckpt_path)

        total_duration = round(time.time() - start_time, 2)
        stats = TrainingStatistics.compute_run_stats(history, total_duration)

        logger.info("Completed training run in %.2f seconds.", total_duration)

        return {
            "history": history,
            "total_duration_seconds": total_duration,
            "final_epoch": self.current_epoch,
            "stats": stats,
        }

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Executes single training epoch over train_dataloader.

        Args:
            epoch: Current 1-indexed epoch integer.

        Returns:
            Dictionary of averaged epoch training metrics.
        """
        return EpochRunner.run_epoch(
            model=self.model,
            dataloader=self.train_dataloader,
            loss_module=self.loss_module,
            optimization_manager=self.optimization_manager,
            device=self.device,
            config=self.config,
            tracker=self.train_tracker,
            scaler=self.scaler,
        )

    def validate(self) -> Dict[str, float]:
        """Executes validation evaluation over val_dataloader.

        Returns:
            Dictionary of averaged validation metrics.
        """
        if self.val_dataloader is None:
            return {}

        return ValidationRunner.run_validation(
            model=self.model,
            dataloader=self.val_dataloader,
            loss_module=self.loss_module,
            device=self.device,
            tracker=self.val_tracker,
        )

    def save_checkpoint(self, checkpoint_path: str) -> None:
        """Saves complete engine state checkpoint file.

        Args:
            checkpoint_path: Target path to save checkpoint file (.pt).
        """
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        opt_state = self.optimization_manager.state_dict()
        model_state = self.model.state_dict()

        checkpoint_data = {
            "model_state_dict": model_state,
            "optimization_state": opt_state,
            "epoch": self.current_epoch,
            "config": self.config,
        }

        torch.save(checkpoint_data, checkpoint_path)
        TrainingLogger.log_checkpoint_saved(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Restores complete engine state from checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint file (.pt).
        """
        if not os.path.exists(checkpoint_path):
            raise EngineValidationError(f"Checkpoint file not found: {checkpoint_path}")

        checkpoint_data = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        if "model_state_dict" in checkpoint_data:
            self.model.load_state_dict(checkpoint_data["model_state_dict"])

        if "optimization_state" in checkpoint_data:
            self.optimization_manager.load_state_dict(checkpoint_data["optimization_state"])

        self.current_epoch = checkpoint_data.get("epoch", 0)
        logger.info("Restored TrainingEngine state from checkpoint: %s (epoch %d)", checkpoint_path, self.current_epoch)
