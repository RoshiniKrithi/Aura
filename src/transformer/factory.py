"""Residual Connection Factory API for Aura LLM Architecture.

Provides central factory method for constructing ResidualConnection modules
from AppConfig or ResidualConfig objects.
"""

import logging
from typing import Any, Optional, Union
import torch

from src.transformer.config import ResidualConfig
from src.transformer.residual import ResidualConnection

logger = logging.getLogger(__name__)


class ResidualFactory:
    """Central factory for constructing and initializing ResidualConnection in Aura."""

    @classmethod
    def create_residual(
        cls,
        config: Optional[Union[ResidualConfig, Any]] = None,
        d_model: Optional[int] = None,
        dropout: Optional[float] = None,
        norm_position: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> ResidualConnection:
        """Instantiates and configures a ResidualConnection module.

        Args:
            config: Optional ResidualConfig or AppConfig object.
            d_model: Optional model feature dimension override.
            dropout: Optional dropout rate override.
            norm_position: Optional norm position override.
            device: Optional target device override.

        Returns:
            Instantiated ResidualConnection module.
        """
        from src.utils.config import AppConfig

        if isinstance(config, AppConfig):
            dim = d_model or config.model.d_model
            drop = dropout if dropout is not None else config.model.dropout
            pos = norm_position or "pre_norm"
            target_device = device or config.system.device
            cfg = ResidualConfig(d_model=dim, dropout=drop, norm_position=pos)
        elif isinstance(config, ResidualConfig):
            cfg = config
            dim = d_model or cfg.d_model
            drop = dropout if dropout is not None else cfg.dropout
            pos = norm_position or cfg.norm_position
            target_device = device or cfg.device
        else:
            cfg = ResidualConfig()
            dim = d_model or cfg.d_model
            drop = dropout if dropout is not None else cfg.dropout
            pos = norm_position or cfg.norm_position
            target_device = device or cfg.device

        module = ResidualConnection(
            config=cfg,
            d_model=dim,
            dropout=drop,
            norm_position=pos,
        )

        if target_device and target_device != "auto":
            try:
                module = module.to(target_device)
                logger.info("Moved ResidualConnection module to target device: %s", target_device)
            except Exception as e:
                logger.warning(
                    "Failed moving ResidualConnection module to device '%s': %s",
                    target_device,
                    str(e),
                )

        return module
