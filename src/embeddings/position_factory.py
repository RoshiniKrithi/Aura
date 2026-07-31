"""Position Embedding Factory for Aura LLM Architecture.

Provides central factory API for constructing Learnable or Sinusoidal position embedding
modules from AppConfig or PositionEmbeddingConfig objects.
"""

import logging
from typing import Any, Optional, Union
import torch
import torch.nn as nn

from src.embeddings.learnable_position import LearnablePositionEmbedding
from src.embeddings.position_config import PositionEmbeddingConfig
from src.embeddings.sinusoidal_position import SinusoidalPositionEmbedding

logger = logging.getLogger(__name__)


class PositionEmbeddingFactory:
    """Central factory for instantiating positional embedding modules in Aura."""

    @classmethod
    def create_position_embedding(
        cls,
        config: Optional[Union[PositionEmbeddingConfig, Any]] = None,
        max_sequence_length: Optional[int] = None,
        d_model: Optional[int] = None,
        embedding_type: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> nn.Module:
        """Instantiates and configures a positional embedding module.

        Args:
            config: Optional PositionEmbeddingConfig or AppConfig object.
            max_sequence_length: Optional max sequence length override M.
            d_model: Optional dense dimension override d.
            embedding_type: Optional type override ("learnable", "sinusoidal").
            device: Optional target device override.

        Returns:
            Instantiated PyTorch nn.Module (LearnablePositionEmbedding or SinusoidalPositionEmbedding).
        """
        from src.utils.config import AppConfig

        if isinstance(config, AppConfig):
            cfg = config.position_embedding
            m_len = max_sequence_length or config.model.max_sequence_length
            dim = d_model or config.model.d_model
            target_device = device or config.system.device
        elif isinstance(config, PositionEmbeddingConfig):
            cfg = config
            m_len = max_sequence_length or cfg.max_sequence_length
            dim = d_model or cfg.d_model
            target_device = device or cfg.device
        else:
            cfg = PositionEmbeddingConfig()
            m_len = max_sequence_length or cfg.max_sequence_length
            dim = d_model or cfg.d_model
            target_device = device or cfg.device

        type_str = (embedding_type or cfg.embedding_type).lower().strip()

        if type_str in ("learnable", "learned", "gpt2"):
            module = LearnablePositionEmbedding(
                config=cfg,
                max_sequence_length=m_len,
                d_model=dim,
            )
        elif type_str in ("sinusoidal", "vaswani", "fixed"):
            module = SinusoidalPositionEmbedding(
                config=cfg,
                max_sequence_length=m_len,
                d_model=dim,
            )
        else:
            raise ValueError(
                f"Unsupported position embedding type: '{type_str}'. "
                f"Supported options: ['learnable', 'sinusoidal']."
            )

        if target_device and target_device != "auto":
            try:
                module = module.to(target_device)
                logger.info("Moved positional embedding module to device: %s", target_device)
            except Exception as e:
                logger.warning(
                    "Failed moving positional embedding module to device '%s': %s",
                    target_device,
                    str(e),
                )

        return module
