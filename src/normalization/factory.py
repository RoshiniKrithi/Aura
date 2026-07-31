"""Layer Normalization Factory API for Aura LLM Architecture.

Provides central factory method for constructing LayerNormalization modules
from AppConfig or LayerNormConfig objects.
"""

import logging
from typing import Any, Optional, Union
import torch

from src.normalization.config import LayerNormConfig
from src.normalization.layer_norm import LayerNormalization

logger = logging.getLogger(__name__)


class LayerNormFactory:
    """Central factory for constructing and initializing LayerNormalization in Aura."""

    @classmethod
    def create_layer_norm(
        cls,
        config: Optional[Union[LayerNormConfig, Any]] = None,
        d_model: Optional[int] = None,
        eps: Optional[float] = None,
        elementwise_affine: Optional[bool] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> LayerNormalization:
        """Instantiates and configures a LayerNormalization module.

        Args:
            config: Optional LayerNormConfig or AppConfig object.
            d_model: Optional model dimension override.
            eps: Optional epsilon constant override.
            elementwise_affine: Optional affine flag override.
            device: Optional target device override.

        Returns:
            Instantiated LayerNormalization module.
        """
        from src.utils.config import AppConfig

        if isinstance(config, AppConfig):
            cfg = config.layernorm
            dim = d_model or config.model.d_model
            eps_val = eps or cfg.eps
            affine = elementwise_affine if elementwise_affine is not None else cfg.elementwise_affine
            target_device = device or config.system.device
        elif isinstance(config, LayerNormConfig):
            cfg = config
            dim = d_model or cfg.d_model
            eps_val = eps or cfg.eps
            affine = elementwise_affine if elementwise_affine is not None else cfg.elementwise_affine
            target_device = device or cfg.device
        else:
            cfg = LayerNormConfig()
            dim = d_model or cfg.d_model
            eps_val = eps or cfg.eps
            affine = elementwise_affine if elementwise_affine is not None else cfg.elementwise_affine
            target_device = device or cfg.device

        module = LayerNormalization(
            config=cfg,
            d_model=dim,
            eps=eps_val,
            elementwise_affine=affine,
        )

        if target_device and target_device != "auto":
            try:
                module = module.to(target_device)
                logger.info("Moved LayerNormalization module to target device: %s", target_device)
            except Exception as e:
                logger.warning("Failed moving LayerNormalization module to device '%s': %s", target_device, str(e))

        return module
