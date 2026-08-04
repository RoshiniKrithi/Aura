"""Checkpoint Configuration Schema for Aura LLM Architecture.

Provides hyperparameter parameters for checkpoint saving, loading, rotation policies,
best model tracking metrics, and model export formats.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class CheckpointConfig:
    """Hyperparameter configuration container for CheckpointManager and LifecycleManager.

    Attributes:
        checkpoint_dir: Directory path for saving checkpoints (default: "checkpoints").
        max_keep_checkpoints: Maximum number of recent checkpoints to retain (default: 5).
        save_best: If True, tracks and saves best performing checkpoint (default: True).
        monitor_metric: Metric name monitored for best model selection (default: "val_loss").
        mode: Optimization direction for monitored metric ("min" or "max", default: "min").
        export_format: Model export format ("pytorch" or "safetensors", default: "pytorch").
    """

    checkpoint_dir: str = "checkpoints"
    max_keep_checkpoints: int = 5
    save_best: bool = True
    monitor_metric: str = "val_loss"
    mode: str = "min"
    export_format: str = "pytorch"


# Alias for backward compatibility
LifecycleConfig = CheckpointConfig
