"""Training Engine Validator for Aura LLM Architecture.

Sanity checks model, DataLoader, loss module, and optimization manager instances
before training execution begins.
"""

from dataclasses import dataclass, field
import logging
from typing import List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.exceptions import EngineValidationError

logger = logging.getLogger(__name__)


@dataclass
class EngineValidationResult:
    """Summary diagnostic report output from TrainingEngine validation checks."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class EngineValidator:
    """Validates TrainingEngine initialization inputs and configurations."""

    @staticmethod
    def validate_setup(
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
    ) -> EngineValidationResult:
        """Sanity checks engine initialization setup components.

        Args:
            model: PyTorch nn.Module instance.
            train_dataloader: Training DataLoader instance.
            val_dataloader: Optional Validation DataLoader instance.

        Returns:
            EngineValidationResult summary object.
        """
        result = EngineValidationResult(is_valid=True)

        if not isinstance(model, nn.Module):
            result.is_valid = False
            result.errors.append(f"model must be a PyTorch nn.Module, got {type(model).__name__}")

        if not isinstance(train_dataloader, DataLoader):
            result.is_valid = False
            result.errors.append(
                f"train_dataloader must be a PyTorch DataLoader, got {type(train_dataloader).__name__}"
            )

        if val_dataloader is not None and not isinstance(val_dataloader, DataLoader):
            result.is_valid = False
            result.errors.append(
                f"val_dataloader must be a PyTorch DataLoader, got {type(val_dataloader).__name__}"
            )

        if not result.is_valid:
            logger.error("Engine setup validation failed: %s", result.errors)

        return result
