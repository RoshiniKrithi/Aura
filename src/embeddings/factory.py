"""Embedding Factory API for Aura LLM Pipeline.

Provides central factory methods for constructing and initializing EmbeddingLayer
instances from AppConfig or EmbeddingConfig objects.
"""

import logging
from typing import Any, Optional, Union
import torch

from src.embeddings.config import EmbeddingConfig
from src.embeddings.embedding_layer import EmbeddingLayer

logger = logging.getLogger(__name__)


class EmbeddingFactory:
    """Central factory for constructing and initializing EmbeddingLayers in Aura architecture."""

    @classmethod
    def create_embedding_layer(
        cls,
        config: Optional[Union[EmbeddingConfig, Any]] = None,
        vocab_size: Optional[int] = None,
        d_model: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> EmbeddingLayer:
        """Instantiates and configures an EmbeddingLayer instance.

        Args:
            config: Optional EmbeddingConfig or AppConfig object.
            vocab_size: Optional vocabulary size override.
            d_model: Optional embedding dimension override.
            device: Optional target device override.

        Returns:
            Instantiated and initialized EmbeddingLayer module.
        """
        from src.utils.config import AppConfig
        if isinstance(config, AppConfig):
            cfg = config.embedding
            v_size = vocab_size or config.model.vocab_size
            dim = d_model or config.model.d_model
            target_device = device or config.system.device
        elif isinstance(config, EmbeddingConfig):
            cfg = config
            v_size = vocab_size or cfg.vocab_size
            dim = d_model or cfg.d_model
            target_device = device or cfg.device
        else:
            cfg = EmbeddingConfig()
            v_size = vocab_size or cfg.vocab_size
            dim = d_model or cfg.d_model
            target_device = device or cfg.device

        layer = EmbeddingLayer(
            config=cfg,
            vocab_size=v_size,
            d_model=dim,
        )

        # Device Placement
        if target_device and target_device != "auto":
            try:
                layer = layer.to(target_device)
                logger.info("Moved EmbeddingLayer to target device: %s", target_device)
            except Exception as e:
                logger.warning("Failed moving EmbeddingLayer to device '%s': %s", target_device, str(e))

        return layer
