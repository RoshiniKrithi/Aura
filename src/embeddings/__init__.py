"""Embeddings Module.

WHY THIS MODULE EXISTS:
    Transformer models require mapping discrete token IDs into continuous vector space
    (Token Embeddings) and injecting sequential order awareness (Positional Embeddings / RoPE).
    This module isolates vector lookup and position encoding logic.

HOW FUTURE MODULES WILL PLUG IN:
    - Phase 4: Will implement custom `TokenEmbedding` PyTorch module.
    - Phase 5: Will implement `PositionalEmbedding` (Learned / Sinusoidal / Rotary Embeddings - RoPE).
    - Phase 11 (Transformer Block) & Phase 12 (GPT Model): The input sequence tokens will first pass through
      `TokenEmbedding` + `PositionalEmbedding` before entering Transformer Blocks.
"""
