"""Transformer Block Factory API for Aura LLM Architecture.

Provides central factory method for constructing TransformerBlock modules
from AppConfig or TransformerBlockConfig objects.
"""

import logging
from typing import Any, Optional, Union
import torch

from src.transformer.block_config import TransformerBlockConfig
from src.transformer.transformer_block import TransformerBlock

logger = logging.getLogger(__name__)


class TransformerBlockFactory:
    """Central factory for constructing and initializing TransformerBlock in Aura."""

    @classmethod
    def create_block(
        cls,
        config: Optional[Union[TransformerBlockConfig, Any]] = None,
        d_model: Optional[int] = None,
        n_heads: Optional[int] = None,
        d_ff: Optional[int] = None,
        dropout: Optional[float] = None,
        activation: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> TransformerBlock:
        """Instantiates and configures a TransformerBlock module.

        Args:
            config: Optional TransformerBlockConfig or AppConfig object.
            d_model: Optional model feature dimension override.
            n_heads: Optional number of attention heads override.
            d_ff: Optional hidden FFN dimension override.
            dropout: Optional dropout probability override.
            activation: Optional activation override ("swiglu" or "gelu").
            device: Optional target device override.

        Returns:
            Instantiated TransformerBlock module.
        """
        from src.utils.config import AppConfig

        if isinstance(config, AppConfig):
            dim = d_model or config.model.d_model
            heads = n_heads or config.model.n_heads
            ff_dim = d_ff or config.model.d_ff
            drop = dropout if dropout is not None else config.model.dropout
            act = activation or config.ffn.activation
            target_device = device or config.system.device
            cfg = TransformerBlockConfig(
                d_model=dim,
                n_heads=heads,
                d_ff=ff_dim,
                dropout=drop,
                activation=act,
                device=target_device,
            )
        elif isinstance(config, TransformerBlockConfig):
            cfg = config
            dim = d_model or cfg.d_model
            heads = n_heads or cfg.n_heads
            ff_dim = d_ff or cfg.d_ff
            drop = dropout if dropout is not None else cfg.dropout
            act = activation or cfg.activation
            target_device = device or cfg.device
        else:
            cfg = TransformerBlockConfig()
            dim = d_model or cfg.d_model
            heads = n_heads or cfg.n_heads
            ff_dim = d_ff or cfg.d_ff
            drop = dropout if dropout is not None else cfg.dropout
            act = activation or cfg.activation
            target_device = device or cfg.device

        module = TransformerBlock(
            config=cfg,
            d_model=dim,
            n_heads=heads,
            d_ff=ff_dim,
            dropout=drop,
            activation=act,
        )

        if target_device and target_device != "auto":
            try:
                module = module.to(target_device)
                logger.info("Moved TransformerBlock module to target device: %s", target_device)
            except Exception as e:
                logger.warning(
                    "Failed moving TransformerBlock module to device '%s': %s",
                    target_device,
                    str(e),
                )

        return module
