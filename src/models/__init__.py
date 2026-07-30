"""Models Module.

WHY THIS MODULE EXISTS:
    Acts as the main architectural registry and entry point for complete model definitions.
    Assembles Token Embeddings, Positional Embeddings, stacked Transformer Blocks, final normalization,
    and Language Model Head (LM Head) into an end-to-end `AuraGPT` PyTorch `nn.Module`.

HOW FUTURE MODULES WILL PLUG IN:
    - Phase 12: Will implement `AuraGPT` top-level model class, weight initialization routines,
      and parameter counter utilities.
    - Phase 13 (Training Loop) & Phase 14 (Inference): Training loops and generation engines will import
      and instantiate `AuraGPT` directly from this module using `ModelConfig`.
"""
