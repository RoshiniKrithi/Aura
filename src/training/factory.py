"""Training Engine Factory API for Aura LLM Architecture.

Provides factory builder methods for constructing TrainingEngine instances
from TrainingEngineConfig or AppConfig.
"""

import logging
from typing import Any, Optional, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.config import TrainingEngineConfig
from src.training.engine import TrainingEngine

logger = logging.getLogger(__name__)


class EngineFactory:
    """Central factory for constructing and configuring TrainingEngine instances."""

    @classmethod
    def create_engine(
        cls,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        config: Optional[Union[TrainingEngineConfig, Any]] = None,
        epochs: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> TrainingEngine:
        """Instantiates and configures a TrainingEngine instance.

        Args:
            model: Target PyTorch nn.Module.
            train_dataloader: Training DataLoader instance.
            val_dataloader: Optional Validation DataLoader instance.
            config: Optional TrainingEngineConfig or AppConfig object.
            epochs: Optional epochs override.
            device: Optional device override.

        Returns:
            Instantiated TrainingEngine orchestrator instance.
        """
        from src.utils.config import AppConfig

        if isinstance(config, AppConfig):
            num_epochs = getattr(config.training, "num_epochs", getattr(config.training, "epochs", 10)) if hasattr(config, "training") else 10
            cfg = TrainingEngineConfig(
                epochs=epochs or num_epochs,
                device=device or (config.system.device if hasattr(config, "system") else "auto"),
                seed=config.system.seed if hasattr(config, "system") else 42,
            )
        elif isinstance(config, TrainingEngineConfig):
            cfg = config
        else:
            cfg = TrainingEngineConfig(
                epochs=epochs or 10,
                device=device or "auto",
            )

        return TrainingEngine(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            config=cfg,
        )
