"""Attention Factory API for Aura LLM Architecture.

Provides central factory method for constructing SelfAttention modules
from AppConfig or AttentionConfig objects.
"""

import logging
from typing import Any, Optional, Union
import torch

from src.attention.config import AttentionConfig
from src.attention.single_head import SelfAttention

logger = logging.getLogger(__name__)


class AttentionFactory:
    """Central factory for constructing and initializing SelfAttention in Aura."""

    @classmethod
    def create_self_attention(
        cls,
        config: Optional[Union[AttentionConfig, Any]] = None,
        d_model: Optional[int] = None,
        d_attn: Optional[int] = None,
        causal: Optional[bool] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> SelfAttention:
        """Instantiates and configures a SelfAttention module.

        Args:
            config: Optional AttentionConfig or AppConfig object.
            d_model: Optional model dimension override.
            d_attn: Optional projection dimension override.
            causal: Optional causal masking override.
            device: Optional target device override.

        Returns:
            Instantiated SelfAttention module.
        """
        from src.utils.config import AppConfig

        if isinstance(config, AppConfig):
            cfg = config.attention
            dim = d_model or config.model.d_model
            attn_dim = d_attn or cfg.d_attn
            target_device = device or config.system.device
        elif isinstance(config, AttentionConfig):
            cfg = config
            dim = d_model or cfg.d_model
            attn_dim = d_attn or cfg.d_attn
            target_device = device or cfg.device
        else:
            cfg = AttentionConfig()
            dim = d_model or cfg.d_model
            attn_dim = d_attn or cfg.d_attn
            target_device = device or cfg.device

        causal_flag = causal if causal is not None else cfg.causal

        module = SelfAttention(
            config=cfg,
            d_model=dim,
            d_attn=attn_dim,
            causal=causal_flag,
        )

        if target_device and target_device != "auto":
            try:
                module = module.to(target_device)
                logger.info("Moved SelfAttention module to target device: %s", target_device)
            except Exception as e:
                logger.warning("Failed moving SelfAttention module to device '%s': %s", target_device, str(e))

        return module

    @classmethod
    def create_multi_head_attention(
        cls,
        config: Optional[Union[AttentionConfig, Any]] = None,
        d_model: Optional[int] = None,
        n_heads: Optional[int] = None,
        causal: Optional[bool] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "MultiHeadAttention":
        """Instantiates and configures a MultiHeadAttention module.

        Args:
            config: Optional AttentionConfig or AppConfig object.
            d_model: Optional model dimension override.
            n_heads: Optional number of heads override.
            causal: Optional causal masking override.
            device: Optional target device override.

        Returns:
            Instantiated MultiHeadAttention module.
        """
        from src.attention.multi_head import MultiHeadAttention
        from src.utils.config import AppConfig

        if isinstance(config, AppConfig):
            cfg = config.attention
            dim = d_model or config.model.d_model
            heads = n_heads or cfg.n_heads
            target_device = device or config.system.device
        elif hasattr(config, "d_model"):
            cfg = config
            dim = d_model or getattr(cfg, "d_model", 768)
            heads = n_heads or getattr(cfg, "n_heads", 12)
            target_device = device or getattr(cfg, "device", "auto")
        else:
            cfg = AttentionConfig()
            dim = d_model or cfg.d_model
            heads = n_heads or cfg.n_heads
            target_device = device or cfg.device

        causal_flag = causal if causal is not None else cfg.causal

        module = MultiHeadAttention(
            d_model=dim,
            n_heads=heads,
            causal=causal_flag,
        )

        if target_device and target_device != "auto":
            try:
                module = module.to(target_device)
                logger.info("Moved MultiHeadAttention module to target device: %s", target_device)
            except Exception as e:
                logger.warning("Failed moving MultiHeadAttention module to device '%s': %s", target_device, str(e))

        return module
