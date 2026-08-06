"""Target Layer Traversal and LoRA Injection Engine for Aura EXP-007.

Provides LoRAInjector for freezing base model parameters and injecting LoRALinear layers.
"""

import logging
from typing import Dict, List, Set, Tuple
import torch
import torch.nn as nn

from src.peft.lora_layer import LoRALinear
from src.peft.peft_config import LoRAConfig

logger = logging.getLogger(__name__)


class LoRAInjector:
    """Freezes base model parameters and injects LoRALinear adapters into target layers."""

    @staticmethod
    def freeze_base_model(model: nn.Module) -> int:
        """Sets requires_grad = False on all base model parameters.

        Returns:
            Total count of frozen parameters.
        """
        frozen_count = 0
        for param in model.parameters():
            param.requires_grad = False
            frozen_count += param.numel()
        return frozen_count

    @classmethod
    def inject_lora(
        cls, model: nn.Module, config: LoRAConfig
    ) -> Tuple[nn.Module, Dict[str, int]]:
        """Traverses model architecture, freezes base weights, and injects LoRALinear layers.

        Args:
            model: PyTorch nn.Module (e.g. AuraGPT).
            config: LoRAConfig specifying rank r, alpha, and target_modules.

        Returns:
            Tuple of (adapted_model, parameter_statistics_dict).
        """
        # 1. Freeze Base Model
        cls.freeze_base_model(model)

        injected_count = 0
        target_set = set(config.target_modules)

        # Recursive module traversal
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Linear):
                # Check if module name contains any target module string
                if any(target in name for target in target_set):
                    parent_name, attr_name = cls._get_parent_name_and_attr(name)
                    parent_module = cls._get_module_by_name(model, parent_name)

                    # Wrap Linear with LoRALinear
                    lora_layer = LoRALinear(
                        base_layer=module,
                        r=config.r,
                        alpha=config.alpha,
                        dropout=config.dropout,
                    )
                    setattr(parent_module, attr_name, lora_layer)
                    injected_count += 1

        # Calculate Parameter Breakdown
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        total_params = trainable_params + frozen_params

        stats = {
            "injected_layers_count": injected_count,
            "trainable_parameters": trainable_params,
            "frozen_parameters": frozen_params,
            "total_parameters": total_params,
            "trainable_percentage": round(trainable_params / max(1, total_params) * 100, 4),
        }

        logger.info(
            "Injected LoRA into %d layers. Trainable Params: %d / %d (%.4f%%)",
            injected_count,
            trainable_params,
            total_params,
            stats["trainable_percentage"],
        )
        return model, stats

    @staticmethod
    def _get_parent_name_and_attr(name: str) -> Tuple[str, str]:
        parts = name.split(".")
        if len(parts) == 1:
            return "", parts[0]
        return ".".join(parts[:-1]), parts[-1]

    @staticmethod
    def _get_module_by_name(model: nn.Module, name: str) -> nn.Module:
        if not name:
            return model
        curr = model
        for part in name.split("."):
            curr = getattr(curr, part)
        return curr

    @staticmethod
    def verify_frozen_parameters(model: nn.Module) -> Tuple[bool, int, int]:
        """Verifies that non-LoRA parameters have requires_grad = False."""
        trainable = 0
        frozen = 0
        base_unfrozen = 0

        for name, param in model.named_parameters():
            if "lora_" in name:
                if param.requires_grad:
                    trainable += param.numel()
            else:
                if param.requires_grad:
                    base_unfrozen += 1
                else:
                    frozen += param.numel()

        is_valid = base_unfrozen == 0
        return is_valid, trainable, frozen
