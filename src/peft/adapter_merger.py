"""Zero-Latency Weight Merger Engine for Aura EXP-007 PEFT.

Provides AdapterMerger for in-place merging of W_merged = W_0 + (alpha / r) * B * A.
"""

import logging
from pathlib import Path
from typing import Union
import torch
import torch.nn as nn

from src.peft.lora_layer import LoRALinear

logger = logging.getLogger(__name__)


class AdapterMerger:
    """Merges LoRA adapter decomposition weights into base model weights for zero-latency inference."""

    @staticmethod
    def merge_adapter_weights(model: nn.Module) -> int:
        """Merges all LoRALinear layers in model tree into base weights in-place.

        Returns:
            Count of merged LoRALinear layers.
        """
        merged_count = 0
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.merge_weights()
                merged_count += 1

        logger.info("Successfully merged %d LoRALinear layers into base model weights.", merged_count)
        return merged_count

    @staticmethod
    def unmerge_adapter_weights(model: nn.Module) -> int:
        """Unmerges all LoRALinear layers in model tree to restore original base weights.

        Returns:
            Count of unmerged LoRALinear layers.
        """
        unmerged_count = 0
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.unmerge_weights()
                unmerged_count += 1

        logger.info("Successfully unmerged %d LoRALinear layers.", unmerged_count)
        return unmerged_count

    @staticmethod
    def export_merged_model(model: nn.Module, output_path: Union[str, Path]) -> Path:
        """Merges adapter weights and saves standalone merged model PyTorch checkpoint.

        Returns:
            Path to exported merged model checkpoint file.
        """
        path = Path(output_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Merge in-place
        AdapterMerger.merge_adapter_weights(model)

        # 2. Extract base state dict
        base_state_dict: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if "lora_" not in name:
                clean_name = name.replace(".base_layer", "")
                base_state_dict[clean_name] = param.cpu().detach().clone()

        checkpoint_data = {
            "model_state_dict": base_state_dict,
            "is_merged_peft": True,
        }
        torch.save(checkpoint_data, path)

        logger.info("Exported standalone merged model checkpoint to %s", path)
        return path
