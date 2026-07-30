"""Transformer Module.

WHY THIS MODULE EXISTS:
    Combines attention mechanisms, normalization layers, feed-forward sub-networks (FFN/MLP),
    and residual connection additions into modular Transformer Decoder blocks.

HOW FUTURE MODULES WILL PLUG IN:
    - Phase 8: Will implement `FeedForward` (SwiGLU/GeLU MLP).
    - Phase 9: Will implement `LayerNorm` / `RMSNorm` from scratch.
    - Phase 10: Will implement `ResidualConnection` wrappers.
    - Phase 11: Will combine Attention + MLP + LayerNorm + Residual into `TransformerBlock`.
    - Phase 12 (GPT Model): The full model will stack `n_layers` instances of `TransformerBlock`.
"""
