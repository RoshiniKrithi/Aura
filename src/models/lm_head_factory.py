"""LM Head Factory API for Aura LLM Architecture.

Provides factory builder methods for constructing LanguageModelingHead instances
from LMHeadConfig, GPTConfig, or AppConfig with optional weight tying.
"""

import logging
from typing import Any, Optional, Union
import torch
import torch.nn as nn

from src.models.config import GPTConfig
from src.models.lm_head import LanguageModelingHead
from src.models.lm_head_config import LMHeadConfig

logger = logging.getLogger(__name__)


class LMHeadFactory:
    """Central factory for constructing and configuring LanguageModelingHead instances."""

    @classmethod
    def create_lm_head(
        cls,
        config: Optional[Union[LMHeadConfig, GPTConfig, Any]] = None,
        tied_weight: Optional[nn.Parameter] = None,
        d_model: Optional[int] = None,
        vocab_size: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> LanguageModelingHead:
        """Instantiates and configures a LanguageModelingHead module.

        Args:
            config: Optional LMHeadConfig, GPTConfig, or AppConfig object.
            tied_weight: Optional nn.Parameter reference for weight tying.
            d_model: Optional d_model override.
            vocab_size: Optional vocab_size override.
            device: Optional target device override.

        Returns:
            Instantiated LanguageModelingHead PyTorch module.
        """
        from src.utils.config import AppConfig

        if isinstance(config, AppConfig):
            cfg = LMHeadConfig(
                d_model=d_model or config.model.d_model,
                vocab_size=vocab_size or config.model.vocab_size,
                bias=config.model.bias,
                device=device or config.system.device,
            )
        elif isinstance(config, GPTConfig):
            cfg = LMHeadConfig(
                d_model=d_model or config.d_model,
                vocab_size=vocab_size or config.vocab_size,
                tie_weights=config.tie_weights,
                bias=config.bias,
                device=device or config.device,
            )
        elif isinstance(config, LMHeadConfig):
            cfg = config
        else:
            cfg = LMHeadConfig(
                d_model=d_model or 768,
                vocab_size=vocab_size or 50257,
            )

        target_device = device or cfg.device

        lm_head = LanguageModelingHead(config=cfg, tied_weight=tied_weight)

        if target_device and target_device != "auto":
            try:
                lm_head = lm_head.to(target_device)
                logger.info("Moved LanguageModelingHead to target device: %s", target_device)
            except Exception as e:
                logger.warning("Failed moving LM Head to device '%s': %s", target_device, str(e))

        return lm_head
