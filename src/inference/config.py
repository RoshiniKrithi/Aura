"""Inference Configuration Schema for Aura LLM Architecture.

Provides hyperparameter parameters for autoregressive text generation,
including max new tokens, temperature scaling, top-k/top-p nucleus filtering,
greedy decoding flags, repetition penalty, and EOS/PAD token IDs.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class InferenceConfig:
    """Hyperparameter configuration container for InferenceEngine.

    Attributes:
        max_new_tokens: Maximum number of new tokens to generate (default: 256).
        temperature: Sampling temperature scalar (default: 0.7).
        top_k: Top-K filtering count (default: 50).
        top_p: Top-P nucleus sampling cumulative probability threshold (default: 0.9).
        do_sample: If False, uses Greedy decoding (default: True).
        repetition_penalty: Multiplier penalty for repeating tokens (default: 1.1).
        eos_token_id: End of sequence token ID (default: 2).
        pad_token_id: Padding token ID (default: 0).
    """

    max_new_tokens: int = 256
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.1
    eos_token_id: int = 2
    pad_token_id: int = 0
