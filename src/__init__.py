"""Aura: Custom GPT-style Programming & DSA Large Language Model.

Root package initialization.

Architecture Overview:
    - tokenizer: Byte-Pair Encoding (BPE) algorithm built from first principles.
    - datasets: Streaming, tokenized dataset pipelines for programming & DSA problems.
    - embeddings: Token embeddings & Rotary Positional Embeddings (RoPE) / Learnable Positional Encodings.
    - attention: Scaled Dot-Product Attention, Causal Masks, & Multi-Head Self-Attention (MHA).
    - transformer: Feed-Forward Networks (SwiGLU/GeLU), RMSNorm/LayerNorm, Residual Blocks.
    - models: Complete Aura GPT model architecture assembly and config bindings.
    - training: Distributed PyTorch training loop, gradient accumulation, and mixed precision.
    - inference: Autoregressive KV-cache text generator, top-k/top-p decoding algorithms.
    - evaluation: Code execution sandbox, HumanEval, MBPP, & DSA benchmark metrics.
    - losses: CrossEntropyLoss with label smoothing & token masking.
    - optimizers: Custom AdamW optimizer implemented from scratch.
    - schedulers: Cosine learning rate scheduler with warmup.
    - utils: Hardware device detection, seeds, paths, configs, & checkpoints.
    - logging: Structured logging framework.
"""

__version__ = "0.1.0"
