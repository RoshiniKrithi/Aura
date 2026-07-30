"""Attention Module.

WHY THIS MODULE EXISTS:
    Self-Attention is the core algorithmic mechanism enabling Transformers to dynamically correlate
    tokens across long code sequences. This module houses scaled dot-product attention calculation,
    causal masking (to enforce autoregressive generation), and Multi-Head Attention (MHA).

HOW FUTURE MODULES WILL PLUG IN:
    - Phase 6: Will implement `ScaledDotProductAttention` with causal triangular masking.
    - Phase 7: Will implement `MultiHeadAttention` parallel projection heads.
    - Phase 11 (Transformer Block): `MultiHeadAttention` will be instantiated inside each Transformer Block layer.
"""
