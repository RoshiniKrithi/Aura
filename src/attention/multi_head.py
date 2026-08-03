"""Production-Grade Multi-Head Causal Self-Attention Implementation for Aura LLM Architecture.

Implements fused single-matrix QKV projection, parallel head splitting and transposition,
4D scaled dot-product attention, lower-triangular causal masking, contiguous head concatenation,
and output projection matching GPT-2/3, LLaMA, and Mistral standards.
"""

import math
import logging
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.attention.exceptions import AttentionValidationError
from src.attention.mask import AttentionMask
from src.attention.multi_head_config import MultiHeadAttentionConfig
from src.attention.validator import AttentionValidator

logger = logging.getLogger(__name__)


class MultiHeadAttention(nn.Module):
    """Production-grade Multi-Head Causal Self-Attention module.

    Design Decisions:
        - Fused c_attn linear projection (d_model -> 3 * d_model) for maximum GPU GEMM efficiency.
        - Efficient 4D tensor reshaping and transposition: (B, T, d) -> (B, H, T, d_head).
        - 4D lower-triangular causal masking broadcasting across batch B and head H dimensions.
        - Contiguous memory layout enforcement before output projection (.contiguous().view).
        - Caches last attention weights tensor (self.last_attention_weights) for multi-head analysis.

    Time Complexity:
        O(B * T * d_model^2 + B * H * T^2 * d_head) projection and matrix multiplication ops.

    Space Complexity:
        O(B * H * T^2) multi-head attention score matrix memory,
        O(4 * d_model^2) projection weight parameters.
    """

    def __init__(
        self,
        config: Optional[MultiHeadAttentionConfig] = None,
        d_model: Optional[int] = None,
        n_heads: Optional[int] = None,
        dropout: float = 0.1,
        causal: bool = True,
        bias: bool = True,
        scale: Optional[float] = None,
    ) -> None:
        """Initializes MultiHeadAttention module.

        Args:
            config: Optional MultiHeadAttentionConfig dataclass.
            d_model: Total embedding/model feature dimension d_model.
            n_heads: Number of parallel attention heads H.
            dropout: Attention weights and residual output dropout probability.
            causal: If True, applies lower-triangular 4D causal mask.
            bias: If True, enables bias terms in linear projections.
            scale: Custom scaling factor (defaults to 1 / sqrt(d_head)).

        Raises:
            ValueError: If d_model is not divisible by n_heads.
        """
        super().__init__()

        cfg = config or MultiHeadAttentionConfig()

        self.d_model = d_model if d_model is not None else cfg.d_model
        self.n_heads = n_heads if n_heads is not None else cfg.n_heads
        self.dropout_p = dropout if config is None else cfg.dropout
        self.causal = causal if config is None else cfg.causal
        self.bias = bias if config is None else cfg.bias

        # Divisibility check: d_model % n_heads == 0
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})."
            )
        self.head_dim = self.d_model // self.n_heads

        # Scale factor: 1 / sqrt(d_head)
        if scale is not None:
            self.scale = scale
        elif cfg.scale is not None:
            self.scale = cfg.scale
        else:
            self.scale = 1.0 / math.sqrt(self.head_dim)

        # 1. Fused QKV Projection Matrix: (d_model -> 3 * d_model)
        self.c_attn = nn.Linear(self.d_model, 3 * self.d_model, bias=self.bias)

        # 2. Output Projection Matrix: (d_model -> d_model)
        self.c_proj = nn.Linear(self.d_model, self.d_model, bias=self.bias)

        # 3. Dropout Layers
        self.attn_dropout = nn.Dropout(p=self.dropout_p)
        self.resid_dropout = nn.Dropout(p=self.dropout_p)

        # 4. Input Validator
        self.validator = AttentionValidator(d_model=self.d_model)

        # Cache for last multi-head attention weights (B, H, T, T)
        self.last_attention_weights: Optional[torch.Tensor] = None

        logger.info(
            "Instantiated MultiHeadAttention: d_model=%d, n_heads=%d, head_dim=%d, Causal=%s",
            self.d_model,
            self.n_heads,
            self.head_dim,
            self.causal,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]]:
        """Forward pass computing Multi-Head Causal Self-Attention.

        Args:
            x: Input FloatTensor of shape (B, T, d_model) or (T, d_model).
            mask: Optional external mask FloatTensor.
            kv_cache: Optional dict containing cached 'key' and 'value' tensors.
            use_cache: If True, returns tuple of (output, new_kv_cache).

        Returns:
            Multi-Head Attention output FloatTensor of shape (B, T, d_model) or (T, d_model),
            or Tuple of (output, updated_kv_cache).

        Raises:
            AttentionValidationError: If input validation checks fail.
        """
        # Validate Input Integrity
        val_res = self.validator.validate_input_embeddings(x)
        if not val_res.is_valid:
            raise AttentionValidationError(f"MultiHeadAttention input validation failed: {val_res.errors}")

        is_2d = x.ndim == 2
        if is_2d:
            x_in = x.unsqueeze(0)  # (T, d) -> (1, T, d)
        else:
            x_in = x

        b_size, seq_len, _ = x_in.shape

        # 1. Fused QKV Projection: (B, T, d_model) -> (B, T, 3 * d_model)
        qkv = self.c_attn(x_in)

        # 2. Split into Query, Key, Value: each of shape (B, T, d_model)
        q, k, v = qkv.split(self.d_model, dim=2)

        # 3. Reshape and Transpose into Parallel Heads: (B, T, d_model) -> (B, H, T, d_head)
        q = q.view(b_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None and "key" in kv_cache and "value" in kv_cache:
            k = torch.cat([kv_cache["key"], k], dim=2)
            v = torch.cat([kv_cache["value"], v], dim=2)

        new_kv_cache = {"key": k, "value": v} if use_cache else None

        # 4. Scaled Dot-Product Attention Scores: (B, H, T, d_head) x (B, H, d_head, T) -> (B, H, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 5. Causal Masking (Broadcasting over B and H)
        if self.causal:
            scores = AttentionMask.apply_causal_mask(scores, custom_mask=mask)
        elif mask is not None:
            scores = scores + mask

        # 6. Softmax Probability Normalization
        attn_weights = F.softmax(scores, dim=-1)

        # Cache last multi-head attention weights for inspection
        self.last_attention_weights = attn_weights.detach()

        # 7. Attention Dropout
        attn_drop = self.attn_dropout(attn_weights)

        # 8. Weighted Value Aggregation: (B, H, T, T) x (B, H, T, d_head) -> (B, H, T, d_head)
        y = torch.matmul(attn_drop, v)

        # 9. Head Concatenation: (B, H, T, d_head) -> (B, T, H, d_head) -> (B, T, d_model)
        y = y.transpose(1, 2).contiguous().view(b_size, -1, self.d_model)

        # 10. Output Linear Projection & Residual Dropout
        output = self.resid_dropout(self.c_proj(y))

        if is_2d:
            output = output.squeeze(0)  # (1, T, d) -> (T, d)

        if use_cache:
            return output, new_kv_cache
        return output

    def extra_repr(self) -> str:
        """Provides readable PyTorch module representation."""
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, head_dim={self.head_dim}, "
            f"causal={self.causal}, bias={self.bias}"
        )
