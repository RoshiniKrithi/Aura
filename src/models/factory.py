"""Model Factory API for Aura LLM Architecture.

Provides central factory methods for constructing GPTModel decoder trunks and AuraGPT models
from AppConfig, GPTConfig, or preset names ("125m", "350m", "1.3b", "7b").
"""

import logging
from typing import Any, Optional, Union
import torch

from src.models.config import GPTConfig
from src.models.gpt import AuraGPT
from src.models.gpt_model import GPTModel

logger = logging.getLogger(__name__)


class ModelFactory:
    """Central factory for constructing and initializing GPTModel decoder trunks in Aura."""

    @classmethod
    def create_gpt_model(
        cls,
        config: Optional[Union[GPTConfig, Any]] = None,
        preset: Optional[str] = None,
        vocab_size: Optional[int] = None,
        d_model: Optional[int] = None,
        n_layers: Optional[int] = None,
        n_heads: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> GPTModel:
        """Instantiates and configures a GPTModel decoder trunk module.

        Args:
            config: Optional GPTConfig or AppConfig object.
            preset: Optional model tier preset ("125m", "350m", "1.3b", "7b").
            vocab_size: Optional vocab size override.
            d_model: Optional d_model override.
            n_layers: Optional n_layers override.
            n_heads: Optional n_heads override.
            device: Optional target device override.

        Returns:
            Instantiated GPTModel PyTorch Module.
        """
        from src.utils.config import AppConfig

        if preset:
            p_clean = preset.lower().strip()
            if "125m" in p_clean:
                cfg = GPTConfig.get_125m_config()
            elif "350m" in p_clean:
                cfg = GPTConfig.get_350m_config()
            elif "1.3b" in p_clean:
                cfg = GPTConfig.get_1_3b_config()
            elif "7b" in p_clean:
                cfg = GPTConfig.get_7b_config()
            else:
                cfg = GPTConfig.get_125m_config()
        elif isinstance(config, AppConfig):
            cfg = GPTConfig(
                model_name=config.model.name,
                vocab_size=vocab_size or config.model.vocab_size,
                max_sequence_length=config.model.max_sequence_length,
                d_model=d_model or config.model.d_model,
                n_layers=n_layers or config.model.n_layers,
                n_heads=n_heads or config.model.n_heads,
                d_ff=config.model.d_ff,
                dropout=config.model.dropout,
                activation=config.ffn.activation,
                norm_type=config.layernorm.norm_type if hasattr(config.layernorm, "norm_type") else "layer_norm",
                eps=config.layernorm.eps,
                bias=config.model.bias,
                initializer_range=config.model.initializer_range,
                device=device or config.system.device,
            )
        elif isinstance(config, GPTConfig):
            cfg = config
        else:
            cfg = GPTConfig()

        target_device = device or cfg.device

        model = GPTModel(config=cfg)

        if target_device and target_device != "auto":
            try:
                model = model.to(target_device)
                logger.info("Moved GPTModel to target device: %s", target_device)
            except Exception as e:
                logger.warning("Failed moving GPTModel to device '%s': %s", target_device, str(e))

        return model


class AuraGPTFactory(ModelFactory):
    """Central factory for constructing and initializing AuraGPT models in Aura."""

    @classmethod
    def create_model(
        cls,
        config: Optional[Union[GPTConfig, Any]] = None,
        preset: Optional[str] = None,
        vocab_size: Optional[int] = None,
        d_model: Optional[int] = None,
        n_layers: Optional[int] = None,
        n_heads: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> AuraGPT:
        """Instantiates and configures an AuraGPT model with LM Head."""
        from src.utils.config import AppConfig

        if preset:
            p_clean = preset.lower().strip()
            if "125m" in p_clean:
                cfg = GPTConfig.get_125m_config()
            elif "350m" in p_clean:
                cfg = GPTConfig.get_350m_config()
            elif "1.3b" in p_clean:
                cfg = GPTConfig.get_1_3b_config()
            elif "7b" in p_clean:
                cfg = GPTConfig.get_7b_config()
            else:
                cfg = GPTConfig.get_125m_config()
        elif isinstance(config, AppConfig):
            cfg = GPTConfig(
                model_name=config.model.name,
                vocab_size=vocab_size or config.model.vocab_size,
                max_sequence_length=config.model.max_sequence_length,
                d_model=d_model or config.model.d_model,
                n_layers=n_layers or config.model.n_layers,
                n_heads=n_heads or config.model.n_heads,
                d_ff=config.model.d_ff,
                dropout=config.model.dropout,
                activation=config.ffn.activation,
                norm_type=config.layernorm.norm_type if hasattr(config.layernorm, "norm_type") else "layer_norm",
                eps=config.layernorm.eps,
                bias=config.model.bias,
                initializer_range=config.model.initializer_range,
                device=device or config.system.device,
            )
        elif isinstance(config, GPTConfig):
            cfg = config
        else:
            cfg = GPTConfig()

        target_device = device or cfg.device

        model = AuraGPT(config=cfg)

        if target_device and target_device != "auto":
            try:
                model = model.to(target_device)
                logger.info("Moved AuraGPT model to target device: %s", target_device)
            except Exception as e:
                logger.warning("Failed moving AuraGPT model to device '%s': %s", target_device, str(e))

        return model
