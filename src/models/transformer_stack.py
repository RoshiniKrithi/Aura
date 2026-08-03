"""Transformer Stack Composition Module for Aura LLM Architecture.

Composes N Transformer Decoder Blocks into a unified sequential execution pipeline.
Handles forward propagation across layers, attention masks, and KV-cache management.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from src.models.config import GPTConfig
from src.transformer.block_config import TransformerBlockConfig
from src.transformer.transformer_block import TransformerBlock

logger = logging.getLogger(__name__)


class TransformerStack(nn.Module):
    """Production-grade Transformer Stack composition module.

    Design Decisions:
        - Composition of N TransformerBlock instances stored in an nn.ModuleList.
        - Sequentially forwards activations through each block.
        - Supports KV-caching across all N layers for autoregressive generation.
    """

    def __init__(self, config: Optional[GPTConfig] = None) -> None:
        """Initializes TransformerStack module.

        Args:
            config: Optional GPTConfig hyperparameter dataclass.
        """
        super().__init__()

        cfg = config or GPTConfig()

        self.d_model = cfg.d_model
        self.n_layers = cfg.n_layers
        self.n_heads = cfg.n_heads
        self.d_ff = cfg.d_ff
        self.dropout_rate = cfg.dropout
        self.activation = cfg.activation

        block_cfg = TransformerBlockConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            dropout=self.dropout_rate,
            activation=self.activation,
            norm_type=cfg.norm_type,
            eps=cfg.eps,
            bias=cfg.bias,
            device=cfg.device,
        )

        self.blocks = nn.ModuleList(
            [TransformerBlock(config=block_cfg) for _ in range(self.n_layers)]
        )

        logger.info("Instantiated TransformerStack with N=%d layers", self.n_layers)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Dict[str, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[List[Dict[str, torch.Tensor]]]]]:
        """Forward pass through N stacked Transformer blocks.

        Args:
            x: Input sequence FloatTensor of shape (B, T, d_model).
            attention_mask: Optional causal/padding mask FloatTensor.
            kv_caches: Optional list of N key/value cache dicts for inference.
            use_cache: If True, returns tuple of (hidden_states, updated_kv_caches).

        Returns:
            Output FloatTensor of shape (B, T, d_model) or Tuple of (hidden_states, updated_kv_caches).
        """
        new_kv_caches: List[Dict[str, torch.Tensor]] = []

        for i, block in enumerate(self.blocks):
            layer_cache = kv_caches[i] if kv_caches is not None else None

            if use_cache or kv_caches is not None:
                x, new_cache = block(
                    x, attention_mask=attention_mask, kv_cache=layer_cache, use_cache=use_cache
                )
                if new_cache is not None:
                    new_kv_caches.append(new_cache)
            else:
                x = block(x, attention_mask=attention_mask)

        if use_cache:
            return x, new_kv_caches
        return x
