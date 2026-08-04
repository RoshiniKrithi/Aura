"""Checkpoint Saver Pipeline for Aura LLM Architecture.

Provides CheckpointSaver for atomic file writes (.tmp -> .pt OS replace),
serializing model weights, optimizer states, scheduler steps, AMP scaler, metadata, and RNG seeds.
"""

import datetime
import hashlib
import logging
import os
import shutil
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn

from src.utils.checkpoint_config import CheckpointConfig
from src.utils.checkpoint_metadata import CheckpointMetadata, MetadataRegistry

logger = logging.getLogger(__name__)


class CheckpointSaver:
    """Executes atomic checkpoint serialization pipeline.

    Time Complexity:
        O(P + S) where P is parameter count and S is optimizer state size.

    Space Complexity:
        O(P + S) memory footprint during serialization.
    """

    def __init__(self, config: Optional[CheckpointConfig] = None) -> None:
        """Initializes CheckpointSaver.

        Args:
            config: Optional CheckpointConfig object.
        """
        self.config = config or CheckpointConfig()
        self.checkpoint_dir = self.config.checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.registry = MetadataRegistry(self.checkpoint_dir)
        self.best_metric_val: Optional[float] = None

    def save(
        self,
        model: nn.Module,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        epoch: int = 0,
        global_step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Any] = None,
    ) -> str:
        """Saves a checkpoint atomically.

        Args:
            model: Target PyTorch nn.Module.
            optimizer: Optional PyTorch Optimizer or OptimizationManager.
            scheduler: Optional PyTorch _LRScheduler.
            scaler: Optional PyTorch GradScaler.
            epoch: Epoch integer.
            global_step: Global step integer.
            metrics: Optional metrics dictionary.
            config: Optional config object.

        Returns:
            Path string to saved checkpoint file (.pt).
        """
        metrics = metrics or {}
        timestamp_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
        filename = f"checkpoint_epoch_{epoch}_step_{global_step}.pt"
        target_path = os.path.join(self.checkpoint_dir, filename)
        tmp_path = f"{target_path}.tmp"

        opt_state = optimizer.state_dict() if optimizer and hasattr(optimizer, "state_dict") else None
        sched_state = scheduler.state_dict() if scheduler and hasattr(scheduler, "state_dict") else None
        scaler_state = scaler.state_dict() if scaler and hasattr(scaler, "state_dict") else None

        rng_states = {
            "torch_cpu_rng": torch.get_rng_state(),
            "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

        payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt_state,
            "scheduler_state_dict": sched_state,
            "scaler_state_dict": scaler_state,
            "epoch": epoch,
            "global_step": global_step,
            "metrics": metrics,
            "config": config or self.config,
            "rng_states": rng_states,
            "timestamp": timestamp_str,
        }

        # Atomic Write (.tmp -> replace)
        torch.save(payload, tmp_path)
        if os.path.exists(tmp_path):
            os.replace(tmp_path, target_path)

        file_size = os.path.getsize(target_path) if os.path.exists(target_path) else 0
        checksum = self._compute_checksum(target_path)

        is_best = self._check_is_best(metrics)
        if is_best and self.config.save_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            shutil.copyfile(target_path, best_path)

        metadata = CheckpointMetadata(
            checkpoint_name=filename,
            epoch=epoch,
            global_step=global_step,
            timestamp=timestamp_str,
            metrics=metrics,
            is_best=is_best,
            file_size_bytes=file_size,
            sha256_checksum=checksum,
        )
        self.registry.add_record(metadata)

        logger.info("CheckpointSaver: Saved atomic checkpoint %s (%d bytes)", target_path, file_size)
        return target_path

    def _check_is_best(self, metrics: Dict[str, float]) -> bool:
        """Determines if current metric improves best metric."""
        metric_name = self.config.monitor_metric
        if metric_name not in metrics:
            return False

        val = metrics[metric_name]
        if self.best_metric_val is None:
            self.best_metric_val = val
            return True

        if self.config.mode == "min":
            if val < self.best_metric_val:
                self.best_metric_val = val
                return True
        else:
            if val > self.best_metric_val:
                self.best_metric_val = val
                return True

        return False

    def _compute_checksum(self, file_path: str) -> str:
        """Calculates SHA256 checksum string of file."""
        if not os.path.exists(file_path):
            return ""
        sha = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                while chunk := f.read(8192):
                    sha.update(chunk)
            return sha.hexdigest()
        except Exception:
            return ""
