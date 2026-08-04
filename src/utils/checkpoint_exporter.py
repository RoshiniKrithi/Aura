"""Model Exporter for Aura LLM Architecture.

Provides utilities for exporting model weights into stripped PyTorch .pt files
or Safetensors distribution formats.
"""

import logging
import os
from typing import Any, Dict
import torch

logger = logging.getLogger(__name__)


class CheckpointExporter:
    """Exports model checkpoints into lightweight distribution formats."""

    @staticmethod
    def export_model_weights(
        checkpoint_path: str,
        output_path: str,
        format: str = "pytorch",
    ) -> str:
        """Extracts model weights from a full checkpoint and exports to target output_path.

        Args:
            checkpoint_path: Input full training checkpoint path (.pt).
            output_path: Destination output file path.
            format: Target format ("pytorch" or "safetensors").

        Returns:
            Path string to exported model weights file.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Input checkpoint file not found: {checkpoint_path}")

        checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_state = checkpoint_data.get("model_state_dict", checkpoint_data.get("model", checkpoint_data))

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        if format.lower() == "safetensors":
            try:
                from safetensors.torch import save_file
                save_file(model_state, output_path)
                logger.info("Exported model weights in Safetensors format to: %s", output_path)
                return output_path
            except ImportError:
                logger.warning("Safetensors package not installed; falling back to PyTorch format.")
                format = "pytorch"

        torch.save(model_state, output_path)
        logger.info("Exported model weights in PyTorch format to: %s", output_path)
        return output_path


# Alias for backward compatibility & requested API
ModelExporter = CheckpointExporter
