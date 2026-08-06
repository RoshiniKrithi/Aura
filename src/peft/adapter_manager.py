"""Adapter Manager, Loader, Saver, Registry, and Exporter for Aura EXP-007 PEFT.

Provides AdapterMetadata, AdapterSaver, AdapterLoader, AdapterSwitcher,
AdapterExporter, AdapterRegistry, and AdapterManager.
"""

from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn

from src.peft.lora_layer import LoRALinear
from src.peft.peft_config import LoRAConfig

logger = logging.getLogger(__name__)


@dataclass
class AdapterMetadata:
    """Dataclass holding standalone adapter metadata."""

    adapter_name: str
    version: str
    rank: int
    alpha: float
    target_modules: List[str]
    trainable_params: int
    base_model_name: str = "AuraGPT"

    def to_dict(self) -> Dict[str, Any]:
        """Converts AdapterMetadata to dictionary representation."""
        return asdict(self)


class AdapterSaver:
    """Saves lightweight standalone adapter state dictionary and metadata."""

    @staticmethod
    def save_adapter(
        model: nn.Module,
        output_dir: Union[str, Path],
        config: LoRAConfig,
        adapter_name: str = "aura_adapter",
        version: str = "v1.0",
    ) -> Path:
        """Exports standalone adapter state dict (lora_A, lora_B parameters only) and config.

        Returns:
            Path to saved adapter directory.
        """
        path = Path(output_dir).resolve() / adapter_name
        path.mkdir(parents=True, exist_ok=True)

        adapter_state_dict: Dict[str, torch.Tensor] = {}
        trainable_count = 0

        for name, param in model.named_parameters():
            if "lora_" in name:
                adapter_state_dict[name] = param.cpu().detach().clone()
                trainable_count += param.numel()

        # Save weights checkpoint (.pt)
        weights_path = path / "adapter_model.pt"
        torch.save(adapter_state_dict, weights_path)

        # Save metadata config (.json)
        metadata = AdapterMetadata(
            adapter_name=adapter_name,
            version=version,
            rank=config.r,
            alpha=config.alpha,
            target_modules=config.target_modules,
            trainable_params=trainable_count,
        )
        config_path = path / "adapter_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        logger.info("Saved standalone adapter %s to %s (%d params)", adapter_name, path, trainable_count)
        return path


class AdapterLoader:
    """Loads standalone adapter weights into target model LoRALinear layers."""

    @staticmethod
    def load_adapter(model: nn.Module, adapter_dir: Union[str, Path]) -> None:
        """Loads adapter state dict into model LoRA parameters."""
        path = Path(adapter_dir).resolve()
        weights_path = path / "adapter_model.pt"

        if not weights_path.exists():
            raise FileNotFoundError(f"Adapter model weights file not found at: {weights_path}")

        adapter_state = torch.load(weights_path, weights_only=True)
        model_state = model.state_dict()

        for name, tensor in adapter_state.items():
            if name in model_state:
                model_state[name].copy_(tensor)

        logger.info("Successfully loaded adapter weights from %s", path.name)


class AdapterSwitcher:
    """Switches active adapter weight sets dynamically on a shared base model."""

    @staticmethod
    def switch_adapter(
        model: nn.Module,
        new_adapter_dir: Union[str, Path],
    ) -> None:
        """Switches model's LoRA parameters to new adapter weights."""
        AdapterLoader.load_adapter(model, new_adapter_dir)


class AdapterRegistry:
    """Central registry tracking available adapter versions and directory paths."""

    def __init__(self) -> None:
        """Initializes AdapterRegistry."""
        self._registry: Dict[str, Path] = {}

    def register_adapter(self, adapter_name: str, adapter_path: Union[str, Path]) -> None:
        """Registers adapter name and directory path."""
        self._registry[adapter_name] = Path(adapter_path).resolve()

    def get_adapter_path(self, adapter_name: str) -> Optional[Path]:
        """Returns directory path for registered adapter name."""
        return self._registry.get(adapter_name)


class AdapterManager:
    """Master manager coordinating adapter saving, loading, switching, and registry."""

    def __init__(self, model: nn.Module) -> None:
        """Initializes AdapterManager."""
        self.model = model
        self.registry = AdapterRegistry()

    def save_adapter(
        self,
        output_dir: Union[str, Path],
        config: LoRAConfig,
        adapter_name: str = "aura_adapter",
        version: str = "v1.0",
    ) -> Path:
        """Saves active model adapter and registers it."""
        path = AdapterSaver.save_adapter(
            model=self.model,
            output_dir=output_dir,
            config=config,
            adapter_name=adapter_name,
            version=version,
        )
        self.registry.register_adapter(adapter_name, path)
        return path

    def load_adapter(self, adapter_path: Union[str, Path]) -> None:
        """Loads adapter weights into model."""
        AdapterLoader.load_adapter(self.model, adapter_path)


class AdapterExporter:
    """Exporter helper for saving and exporting standalone adapters."""

    @staticmethod
    def export_adapter(
        model: nn.Module,
        output_dir: Union[str, Path],
        config: LoRAConfig,
        adapter_name: str = "aura_adapter",
        version: str = "v1.0",
    ) -> Path:
        """Exports standalone adapter files."""
        return AdapterSaver.save_adapter(
            model=model,
            output_dir=output_dir,
            config=config,
            adapter_name=adapter_name,
            version=version,
        )

