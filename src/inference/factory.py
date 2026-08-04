"""Inference Factory API for Aura LLM Architecture.

Provides factory builder methods for constructing InferenceEngine instances
from InferenceConfig or AppConfig.
"""

import logging
from typing import Any, Optional, Union
import torch
import torch.nn as nn

from src.inference.config import InferenceConfig
from src.inference.engine import InferenceEngine

logger = logging.getLogger(__name__)


class InferenceFactory:
    """Central factory for constructing and configuring InferenceEngine instances."""

    @classmethod
    def create_engine(
        cls,
        model: nn.Module,
        tokenizer: Any,
        config: Optional[Union[InferenceConfig, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> InferenceEngine:
        """Instantiates and configures an InferenceEngine instance.

        Args:
            model: Target PyTorch nn.Module.
            tokenizer: Tokenizer instance.
            config: Optional InferenceConfig or AppConfig object.
            device: Optional device override.

        Returns:
            Instantiated InferenceEngine instance.
        """
        from src.utils.config import AppConfig

        if isinstance(config, AppConfig):
            cfg = InferenceConfig(
                max_new_tokens=config.inference.max_new_tokens if hasattr(config, "inference") else 256,
                temperature=config.inference.temperature if hasattr(config, "inference") else 0.7,
                top_k=config.inference.top_k if hasattr(config, "inference") else 50,
                top_p=config.inference.top_p if hasattr(config, "inference") else 0.9,
            )
        elif isinstance(config, InferenceConfig):
            cfg = config
        else:
            cfg = InferenceConfig()

        return InferenceEngine(
            model=model,
            tokenizer=tokenizer,
            config=cfg,
            device=device,
        )
