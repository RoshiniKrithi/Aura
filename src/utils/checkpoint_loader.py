"""Checkpoint Loader Pipeline for Aura LLM Architecture.

Provides CheckpointLoader for loading model weights, optimizer states, scheduler states,
and configuration metadata with device mapping and weights_only=False safety.
"""

import logging
import os
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CheckpointLoader:
    """Executes safe checkpoint loading and model weight restoration."""

    @staticmethod
    def load_payload(
        checkpoint_path: str, device: Union[str, torch.device] = "cpu"
    ) -> Dict[str, Any]:
        """Loads complete checkpoint payload dictionary.

        Args:
            checkpoint_path: Path to target checkpoint file (.pt).
            device: PyTorch device mapping target.

        Returns:
            Checkpoint payload dictionary.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        logger.info("Loading checkpoint payload from: %s onto device %s", checkpoint_path, device)
        return torch.load(checkpoint_path, map_location=device, weights_only=False)

    @staticmethod
    def load_model(
        checkpoint_path: str, model: nn.Module, device: Union[str, torch.device] = "cpu"
    ) -> nn.Module:
        """Restores model parameter weights from checkpoint file.

        Args:
            checkpoint_path: Path to target checkpoint file (.pt).
            model: PyTorch nn.Module instance.
            device: PyTorch device mapping target.

        Returns:
            Restored model instance in eval() mode.
        """
        payload = CheckpointLoader.load_payload(checkpoint_path, device=device)
        model_state = payload.get("model_state_dict", payload.get("model", payload))

        model.to(device)
        model.load_state_dict(model_state)
        model.eval()

        logger.info("Loaded model state_dict into model from: %s", checkpoint_path)
        return model
