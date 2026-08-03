"""Production-Grade Transformer Decoder Block Module for Aura LLM Architecture.

Assembles Pre-LN LayerNormalization, Causal Multi-Head Self-Attention,
Feed-Forward Network (SwiGLU / GeLU), and Residual skip connections into a unified block.
"""

import logging
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

from src.attention.multi_head import MultiHeadAttention
from src.attention.multi_head_config import MultiHeadAttentionConfig
from src.ffn.config import FeedForwardConfig
from src.ffn.network import FeedForwardNetwork
from src.normalization.config import LayerNormConfig
from src.normalization.layer_norm import LayerNormalization
from src.transformer.block_config import TransformerBlockConfig
from src.transformer.block_validator import (
    TransformerBlockValidationError,
    TransformerBlockValidator,
)
from src.transformer.config import ResidualConfig
from src.transformer.residual import ResidualConnection

logger = logging.getLogger(__name__)


class TransformerBlock(nn.Module):
    """Production-grade GPT Transformer Decoder Block module.

    Pre-LN Architecture:
        1. x_norm1 = LN_1(x)
        2. attn_out = MultiHeadAttention(x_norm1, mask)
        3. x_mha = x + Dropout(attn_out)
        4. x_norm2 = LN_2(x_mha)
        5. ffn_out = FeedForward(x_norm2)
        6. x_out = x_mha + Dropout(ffn_out)

    Time Complexity:
        O(B * T * d_model^2 + B * T^2 * d_model) attention + FFN computation per block.

    Space Complexity:
        O(B * T * d_model) activation memory.
    """

    def __init__(
        self,
        config: Optional[TransformerBlockConfig] = None,
        d_model: Optional[int] = None,
        n_heads: Optional[int] = None,
        d_ff: Optional[int] = None,
        dropout: Optional[float] = None,
        activation: Optional[str] = None,
    ) -> None:
        """Initializes TransformerBlock module and sub-modules.

        Args:
            config: Optional TransformerBlockConfig dataclass.
            d_model: Optional model dimension override.
            n_heads: Optional number of attention heads override.
            d_ff: Optional hidden FFN dimension override.
            dropout: Optional dropout rate override.
            activation: Optional activation function override ("swiglu" or "gelu").
        """
        super().__init__()

        cfg = config or TransformerBlockConfig()

        self.d_model = d_model if d_model is not None else cfg.d_model
        self.n_heads = n_heads if n_heads is not None else cfg.n_heads
        self.d_ff = d_ff if d_ff is not None else cfg.d_ff
        self.dropout_rate = dropout if dropout is not None else cfg.dropout
        self.activation_type = activation if activation is not None else cfg.activation

        # 1. LayerNormalization Sub-Modules (Pre-LN 1 and Pre-LN 2)
        ln_cfg = cfg.norm_config or LayerNormConfig(
            d_model=self.d_model, eps=cfg.eps, device=cfg.device
        )
        self.ln_1 = LayerNormalization(config=ln_cfg)
        self.ln_2 = LayerNormalization(config=ln_cfg)

        # 2. Causal Multi-Head Self-Attention Sub-Module
        mha_cfg = cfg.attn_config or MultiHeadAttentionConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            dropout=self.dropout_rate,
            bias=cfg.bias,
        )
        self.attn = MultiHeadAttention(config=mha_cfg)

        # 3. Feed-Forward Network Sub-Module (SwiGLU / GeLU)
        ffn_cfg = cfg.ffn_config or FeedForwardConfig(
            d_model=self.d_model,
            d_ff=self.d_ff,
            activation=self.activation_type,
            dropout=self.dropout_rate,
            bias=cfg.bias,
        )
        self.ffn = FeedForwardNetwork(config=ffn_cfg)

        # 4. Residual Connection Sub-Modules (Res 1 and Res 2)
        res_cfg = cfg.residual_config or ResidualConfig(
            d_model=self.d_model, dropout=self.dropout_rate
        )
        self.res_1 = ResidualConnection(config=res_cfg)
        self.res_2 = ResidualConnection(config=res_cfg)

        # 5. Input Validator
        self.validator = TransformerBlockValidator(d_model=self.d_model)

        # Diagnostic state caches
        self.last_attn_norm: Optional[float] = None
        self.last_ffn_norm: Optional[float] = None

        logger.info(
            "Instantiated TransformerBlock: d_model=%d, n_heads=%d, d_ff=%d, activation=%s, dropout=%.2f",
            self.d_model,
            self.n_heads,
            self.d_ff,
            self.activation_type,
            self.dropout_rate,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]]:
        """Forward pass through Pre-LN Transformer Decoder Block.

        Args:
            x: Input FloatTensor of shape (B, T, d_model) or (T, d_model).
            attention_mask: Optional causal/padding mask FloatTensor.
            kv_cache: Optional cached key/value states for autoregressive inference.
            use_cache: If True, returns tuple of (output_tensor, updated_kv_cache).

        Returns:
            Output FloatTensor of shape (B, T, d_model) or Tuple of (Output, updated_kv_cache).

        Raises:
            TransformerBlockValidationError: If validation fails.
        """
        # Validate Input Integrity
        val_res = self.validator.validate_inputs(x, attention_mask=attention_mask)
        if not val_res.is_valid:
            raise TransformerBlockValidationError(
                f"TransformerBlock input validation failed: {val_res.errors}"
            )

        # 1. Pre-LN 1 -> Multi-Head Self-Attention -> Residual 1
        x_norm1 = self.ln_1(x)

        if use_cache or kv_cache is not None:
            attn_out, new_kv_cache = self.attn(
                x_norm1, mask=attention_mask, kv_cache=kv_cache, use_cache=use_cache
            )
        else:
            attn_out = self.attn(x_norm1, mask=attention_mask)
            new_kv_cache = None

        x_mha = self.res_1(x, attn_out)

        # 2. Pre-LN 2 -> Feed-Forward Network -> Residual 2
        x_norm2 = self.ln_2(x_mha)
        ffn_out = self.ffn(x_norm2)
        x_out = self.res_2(x_mha, ffn_out)

        # Cache diagnostic norms
        with torch.no_grad():
            self.last_attn_norm = torch.norm(attn_out.detach(), p=2).item()
            self.last_ffn_norm = torch.norm(ffn_out.detach(), p=2).item()

        if use_cache:
            return x_out, new_kv_cache
        return x_out

    def extra_repr(self) -> str:
        """Provides readable PyTorch module representation."""
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, d_ff={self.d_ff}, "
            f"activation='{self.activation_type}', dropout={self.dropout_rate}"
        )
