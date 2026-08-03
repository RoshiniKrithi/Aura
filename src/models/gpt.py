"""Production-Grade AuraGPT Decoder Model Implementation for Aura Architecture.

Assembles Token Embeddings, Positional Embeddings, stacked Transformer Blocks (Pre-LN),
final Layer Normalization, and Language Modeling Projection Head matching GPT-2/3/4 standards.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.embeddings.embedding_layer import EmbeddingLayer
from src.embeddings.sinusoidal_position import SinusoidalPositionEmbedding
from src.models.config import AuraGPTConfig
from src.models.exceptions import ModelValidationError
from src.models.initializer import ModelInitializer
from src.models.validator import ModelValidator
from src.normalization.layer_norm import LayerNormalization
from src.transformer.block_config import TransformerBlockConfig
from src.transformer.transformer_block import TransformerBlock

logger = logging.getLogger(__name__)


class AuraGPT(nn.Module):
    """Production-grade GPT Decoder-Only Large Language Model.

    Pipeline:
        1. Token Embedding Lookup: (B, T) -> (B, T, d_model)
        2. Positional Embedding Addition: (B, T, d_model) + (1, T, d_model)
        3. Embedding Dropout Regularization
        4. Stack of N Transformer Blocks with Pre-LN Attention and Feed-Forward Networks
        5. Final Layer Normalization
        6. LM Head Projection: (B, T, d_model) -> (B, T, vocab_size)

    Time Complexity:
        O(N * (B * T * d_model^2 + B * T^2 * d_model) + B * T * d_model * vocab_size).

    Space Complexity:
        O(B * T * d_model) activation memory across N blocks,
        O(vocab_size * d_model + N * params_per_block) parameters.
    """

    def __init__(self, config: Optional[AuraGPTConfig] = None) -> None:
        """Initializes AuraGPT sub-modules and weight initializations.

        Args:
            config: Optional AuraGPTConfig hyperparameter object.
        """
        super().__init__()

        cfg = config or AuraGPTConfig()

        self.config = cfg
        self.model_name = cfg.model_name
        self.vocab_size = cfg.vocab_size
        self.max_sequence_length = cfg.max_sequence_length
        self.d_model = cfg.d_model
        self.n_layers = cfg.n_layers
        self.n_heads = cfg.n_heads
        self.d_ff = cfg.d_ff
        self.dropout_rate = cfg.dropout
        self.activation = cfg.activation
        self.tie_weights = cfg.tie_weights

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

        # 4. Stack of N Transformer Decoder Blocks
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

        # 5. Final Layer Normalization
        self.ln_f = LayerNormalization(d_model=self.d_model, eps=cfg.eps)

        # 6. Language Modeling Head (LM Head)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)

        # Weight Tying: Share weights between Token Embeddings and LM Head
        if self.tie_weights:
            self.lm_head.weight = self.tok_embeddings.weight

        # 7. Input & Sequence Validator
        self.validator = ModelValidator(
            vocab_size=self.vocab_size, max_sequence_length=self.max_sequence_length
        )

        # 8. Scaled Weight Initializations
        ModelInitializer.initialize_weights(
            self, initializer_range=cfg.initializer_range, n_layers=self.n_layers
        )

        logger.info(
            "Instantiated AuraGPT model '%s': params=%d, d_model=%d, n_layers=%d, n_heads=%d, vocab_size=%d, tie_weights=%s",
            self.model_name,
            self.get_num_params(),
            self.d_model,
            self.n_layers,
            self.n_heads,
            self.vocab_size,
            self.tie_weights,
        )

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Calculates total trainable parameters in the model.

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
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Dict[str, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]]:
        """Forward pass through AuraGPT model.

        Args:
            input_ids: Input token ID LongTensor of shape (B, T) or (T,).
            targets: Optional target token ID LongTensor of shape (B, T) for loss calculation.
            kv_caches: Optional list of N key/value cache dicts for autoregressive inference.
            use_cache: If True, returns updated KV caches list for token generation.

        Returns:
            - If targets is provided: Tuple of (logits, loss).
            - If use_cache is True: Tuple of (logits, new_kv_caches).
            - Otherwise: Unnormalized Logits FloatTensor of shape (B, T, vocab_size).

        Raises:
            ModelValidationError: If input validation fails.
        """
        # Validate Input Integrity
        val_res = self.validator.validate_inputs(input_ids, targets=targets)
        if not val_res.is_valid:
            raise ModelValidationError(f"AuraGPT input validation failed: {val_res.errors}")

        is_1d = input_ids.ndim == 1
        if is_1d:
            input_ids_in = input_ids.unsqueeze(0)  # (T,) -> (1, T)
            if targets is not None:
                targets = targets.unsqueeze(0)
        else:
            input_ids_in = input_ids

        b_size, seq_len = input_ids_in.shape

        # 1. Compute Token & Positional Embeddings
        tok_emb = self.tok_embeddings(input_ids_in)  # (B, T, d_model)
        pos_emb = self.pos_embeddings(sequence_length=seq_len, batch_size=b_size)  # (B, T, d_model)
        x = self.emb_dropout(tok_emb + pos_emb)

        # 2. Pass Through Stack of N Transformer Decoder Blocks
        new_kv_caches: List[Dict[str, torch.Tensor]] = []

        for i, block in enumerate(self.blocks):
            block_cache = kv_caches[i] if kv_caches is not None else None

            if use_cache or kv_caches is not None:
                x, new_cache = block(x, kv_cache=block_cache, use_cache=use_cache)
                if new_cache is not None:
                    new_kv_caches.append(new_cache)
            else:
                x = block(x)

        # 3. Final Layer Normalization
        x = self.ln_f(x)  # (B, T, d_model)

        # 4. Language Modeling Head Projection
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if is_1d:
            logits = logits.squeeze(0)  # (1, T, vocab_size) -> (T, vocab_size)

        # 5. Training Loss Computation (Cross-Entropy Loss)
        if targets is not None:
            # Shift logits and targets for next-token prediction if needed or align shapes
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-1,
            )
            return logits, loss

        if use_cache:
            return logits, new_kv_caches

        return logits

    def extra_repr(self) -> str:
        """Provides readable PyTorch module representation."""
        return (
            f"model_name='{self.model_name}', params={self.get_num_params()}, "
            f"vocab_size={self.vocab_size}, d_model={self.d_model}, n_layers={self.n_layers}, "
            f"n_heads={self.n_heads}, tie_weights={self.tie_weights}"
        )
