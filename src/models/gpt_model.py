"""Production-Grade GPT Model Decoder Trunk for Aura LLM Architecture.

Implements input token embeddings, positional embeddings, embedding dropout,
stacked Transformer decoder blocks (TransformerStack), and final Layer Normalization.
Returns standardized hidden states without LM Head projection, logits, or loss calculation.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from src.embeddings.embedding_layer import EmbeddingLayer
from src.embeddings.sinusoidal_position import SinusoidalPositionEmbedding
from src.models.config import GPTConfig
from src.models.exceptions import ModelValidationError
from src.models.initializer import ModelInitializer
from src.models.transformer_stack import TransformerStack
from src.models.validator import ModelValidator
from src.normalization.layer_norm import LayerNormalization

logger = logging.getLogger(__name__)


class GPTModel(nn.Module):
    """Production-grade GPT Decoder Trunk module.

    Pipeline:
        1. Token Embedding Lookup: (B, T) -> (B, T, d_model)
        2. Positional Embedding Addition: (B, T, d_model) + (1, T, d_model)
        3. Embedding Dropout Regularization
        4. Pass through TransformerStack (N x Transformer Blocks)
        5. Final Layer Normalization: (B, T, d_model) -> (B, T, d_model)
        6. Return Final Hidden States

    Time Complexity:
        O(N * (B * T * d_model^2 + B * T^2 * d_model)).

    Space Complexity:
        O(B * T * d_model) activation memory across N blocks.
    """

    def __init__(self, config: Optional[GPTConfig] = None) -> None:
        """Initializes GPTModel decoder trunk.

        Args:
            config: Optional GPTConfig hyperparameter dataclass.
        """
        super().__init__()

        cfg = config or GPTConfig()

        self.config = cfg
        self.model_name = cfg.model_name
        self.vocab_size = cfg.vocab_size
        self.max_sequence_length = cfg.max_sequence_length
        self.d_model = cfg.d_model
        self.n_layers = cfg.n_layers
        self.n_heads = cfg.n_heads
        self.d_ff = cfg.d_ff
        self.dropout_rate = cfg.dropout

        # 1. Token Embeddings Layer
        self.tok_embeddings = EmbeddingLayer(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
        )

        # 2. Positional Embeddings Layer
        self.pos_embeddings = SinusoidalPositionEmbedding(
            d_model=self.d_model,
            max_sequence_length=self.max_sequence_length,
        )

        # 3. Embedding Dropout
        self.emb_dropout = nn.Dropout(p=self.dropout_rate)

        # 4. Stack of N Transformer Decoder Blocks (TransformerStack Composition)
        self.transformer_stack = TransformerStack(config=cfg)

        # 5. Final Layer Normalization
        self.ln_f = LayerNormalization(d_model=self.d_model, eps=cfg.eps)

        # 6. Input Validator
        self.validator = ModelValidator(
            vocab_size=self.vocab_size, max_sequence_length=self.max_sequence_length
        )

        # 7. Scaled Weight Initialization
        ModelInitializer.initialize_weights(
            self, initializer_range=cfg.initializer_range, n_layers=self.n_layers
        )

        # Cache for last hidden states output
        self.last_hidden_states: Optional[torch.Tensor] = None

        logger.info(
            "Instantiated GPTModel trunk '%s': params=%d, d_model=%d, n_layers=%d, n_heads=%d, vocab_size=%d",
            self.model_name,
            self.get_num_params(),
            self.d_model,
            self.n_layers,
            self.n_heads,
            self.vocab_size,
        )

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Calculates total trainable parameters in the decoder trunk.

        Args:
            non_embedding: If True, excludes positional embedding parameters.

        Returns:
            Integer parameter count.
        """
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding and hasattr(self.pos_embeddings, "parameters"):
            pos_params = sum(p.numel() for p in self.pos_embeddings.parameters() if p.requires_grad)
            n_params -= pos_params
        return n_params

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Dict[str, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[List[Dict[str, torch.Tensor]]]]]:
        """Forward pass through GPTModel returning final normalized hidden states.

        Args:
            input_ids: Input token ID LongTensor of shape (B, T) or (T,).
            attention_mask: Optional causal/padding mask FloatTensor.
            kv_caches: Optional list of N key/value cache dicts for autoregressive inference.
            use_cache: If True, returns tuple of (hidden_states, updated_kv_caches).

        Returns:
            Hidden States FloatTensor of shape (B, T, d_model) or (T, d_model),
            or Tuple of (hidden_states, updated_kv_caches).

        Raises:
            ModelValidationError: If input validation checks fail.
        """
        # Validate Input Integrity
        val_res = self.validator.validate_inputs(input_ids)
        if not val_res.is_valid:
            raise ModelValidationError(f"GPTModel input validation failed: {val_res.errors}")

        is_1d = input_ids.ndim == 1
        if is_1d:
            input_ids_in = input_ids.unsqueeze(0)  # (T,) -> (1, T)
        else:
            input_ids_in = input_ids

        b_size, seq_len = input_ids_in.shape

        # 1. Compute Token & Positional Embeddings
        tok_emb = self.tok_embeddings(input_ids_in)  # (B, T, d_model)
        pos_emb = self.pos_embeddings(sequence_length=seq_len, batch_size=b_size)  # (B, T, d_model)
        x = self.emb_dropout(tok_emb + pos_emb)

        # 2. Pass Through TransformerStack
        if use_cache or kv_caches is not None:
            x, new_kv_caches = self.transformer_stack(
                x, attention_mask=attention_mask, kv_caches=kv_caches, use_cache=use_cache
            )
        else:
            x = self.transformer_stack(x, attention_mask=attention_mask)
            new_kv_caches = None

        # 3. Final Layer Normalization
        hidden_states = self.ln_f(x)  # (B, T, d_model)

        # Cache last hidden states for inspection
        self.last_hidden_states = hidden_states.detach()

        if is_1d:
            hidden_states = hidden_states.squeeze(0)  # (1, T, d_model) -> (T, d_model)

        if use_cache:
            return hidden_states, new_kv_caches

        return hidden_states

    def extra_repr(self) -> str:
        """Provides readable PyTorch module representation."""
        return (
            f"model_name='{self.model_name}', params={self.get_num_params()}, "
            f"vocab_size={self.vocab_size}, d_model={self.d_model}, n_layers={self.n_layers}, "
            f"n_heads={self.n_heads}"
        )
