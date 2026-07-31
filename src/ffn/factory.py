"""Feed Forward Factory API for Aura LLM Architecture.

Provides central factory method for constructing FeedForwardNetwork modules
from AppConfig or FeedForwardConfig objects.
"""

import logging
from typing import Any, Optional, Union
import torch

from src.ffn.config import FeedForwardConfig
from src.ffn.network import FeedForwardNetwork

logger = logging.getLogger(__name__)


class FeedForwardFactory:
    """Central factory for constructing and initializing FeedForwardNetwork in Aura."""

    @classmethod
    def create_feed_forward(
        cls,
        config: Optional[Union[FeedForwardConfig, Any]] = None,
        d_model: Optional[int] = None,
        d_ff: Optional[int] = None,
        activation: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> FeedForwardNetwork:
        """Instantiates and configures a FeedForwardNetwork module.

        Args:
            config: Optional FeedForwardConfig or AppConfig object.
            d_model: Optional model dimension override.
            d_ff: Optional hidden expansion dimension override.
            activation: Optional activation function override.
            device: Optional target device override.

        Returns:
            Instantiated FeedForwardNetwork module.
        """
        from src.utils.config import AppConfig

        if isinstance(config, AppConfig):
            cfg = config.ffn
            dim = d_model or config.model.d_model
            hidden_dim = d_ff or cfg.hidden_dim
            act = activation or cfg.activation
            target_device = device or config.system.device
        elif isinstance(config, FeedForwardConfig):
            cfg = config
            dim = d_model or cfg.d_model
            hidden_dim = d_ff or cfg.hidden_dim
            act = activation or cfg.activation
            target_device = device or cfg.device
        else:
            cfg = FeedForwardConfig()
            dim = d_model or cfg.d_model
            hidden_dim = d_ff or cfg.hidden_dim
            act = activation or cfg.activation
            target_device = device or cfg.device

        module = FeedForwardNetwork(
            config=cfg,
            d_model=dim,
            d_ff=hidden_dim,
            activation=act,
        )

        if target_device and target_device != "auto":
            try:
                module = module.to(target_device)
                logger.info("Moved FeedForwardNetwork module to target device: %s", target_device)
            except Exception as e:
                logger.warning("Failed moving FeedForwardNetwork module to device '%s': %s", target_device, str(e))

        return module
