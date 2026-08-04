"""Inference Helper Utilities for Aura LLM Architecture.

Provides repetition penalty processing and prompt context window truncation helpers.
"""

import logging
import torch

logger = logging.getLogger(__name__)


class InferenceUtilities:
    """Helper utility routines for autoregressive generation."""

    @staticmethod
    def apply_repetition_penalty(
        logits: torch.Tensor, generated_tokens: torch.Tensor, penalty: float = 1.1
    ) -> torch.Tensor:
        """Applies multiplicative repetition penalty to previously generated token logits.

        Args:
            logits: Logit tensor of shape (batch_size, vocab_size).
            generated_tokens: Token ID history tensor of shape (batch_size, seq_len).
            penalty: Multiplicative penalty factor (default: 1.1).

        Returns:
            Penalized logits tensor.
        """
        if penalty == 1.0 or generated_tokens.numel() == 0:
            return logits

        logits = logits.clone()
        for batch_idx in range(logits.size(0)):
            tokens = generated_tokens[batch_idx].unique()
            for token_id in tokens:
                token_val = token_id.item()
                if token_val < logits.size(-1):
                    if logits[batch_idx, token_val] < 0:
                        logits[batch_idx, token_val] *= penalty
                    else:
                        logits[batch_idx, token_val] /= penalty

        return logits

    @staticmethod
    def truncate_prompt_tokens(
        prompt_tokens: torch.Tensor, max_allowed_tokens: int = 1024
    ) -> torch.Tensor:
        """Truncates prompt tokens from the left if length exceeds max_allowed_tokens.

        Args:
            prompt_tokens: Input token tensor of shape (batch_size, prompt_len).
            max_allowed_tokens: Maximum allowed token sequence length.

        Returns:
            Truncated token tensor of shape (batch_size, min(prompt_len, max_allowed_tokens)).
        """
        if prompt_tokens.size(-1) <= max_allowed_tokens:
            return prompt_tokens

        logger.warning(
            "Prompt length (%d) exceeds max allowed (%d); truncating context from left.",
            prompt_tokens.size(-1),
            max_allowed_tokens,
        )
        return prompt_tokens[:, -max_allowed_tokens:]
