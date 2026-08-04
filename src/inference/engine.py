"""Production-Grade Inference & Autoregressive Text Generation Engine for Aura LLM.

Orchestrates prompt tokenization, context window truncation, autoregressive forward passes,
sampling strategy pipeline execution, repetition penalty, stop token checks, and text decoding.
"""

import time
import logging
from typing import Any, Dict, Generator, Optional, Union
import torch
import torch.nn as nn

from src.inference.config import InferenceConfig
from src.inference.exceptions import InferenceValidationError
from src.inference.statistics import InferenceStatistics, InferenceStats
from src.inference.strategies import CompositeSamplingStrategy
from src.inference.utilities import InferenceUtilities
from src.inference.validator import InferenceValidator

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Production-grade Autoregressive Text Generation Engine.

    Design Decisions:
        - Decouples decoding strategy filters from model trunk via Strategy Pattern.
        - Supports Greedy Decoding, Temperature Scaling, Top-K Filtering, and Top-P (Nucleus) Sampling.
        - Supports token-by-token streaming generation.
        - Enforces context window bounds and stopping criteria (EOS token ID or max new tokens).
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: Optional[InferenceConfig] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Initializes InferenceEngine binding model and tokenizer.

        Args:
            model: PyTorch nn.Module instance.
            tokenizer: Tokenizer instance implementing encode() and decode() methods.
            config: Optional InferenceConfig hyperparameter object.
            device: Optional target device ("cpu", "cuda", etc.).
        """
        val_res = InferenceValidator.validate_setup(model, tokenizer)
        if not val_res.is_valid:
            raise InferenceValidationError(f"Inference setup validation failed: {val_res.errors}")

        self.config = config or InferenceConfig()
        self.model = model
        self.tokenizer = tokenizer

        # Device Setup
        if device is not None:
            self.device = torch.device(device)
        elif next(model.parameters(), None) is not None:
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device("cpu")

        try:
            self.model = self.model.to(self.device)
        except Exception as e:
            logger.warning("Moving model to device '%s' failed: %s", self.device, str(e))

        self.composite_strategy = CompositeSamplingStrategy()

        logger.info("Instantiated InferenceEngine on device '%s'.", self.device)

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
    ) -> str:
        """Generates full text completion for a given prompt string.

        Args:
            prompt: Input text prompt string.
            max_new_tokens: Optional max new tokens override.
            temperature: Optional temperature override.
            top_k: Optional top_k override.
            top_p: Optional top_p override.
            do_sample: Optional sampling flag override.

        Returns:
            Generated text string response (prompt + generated response).
        """
        cfg = self._override_config(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
        )

        # 1. Encode Prompt
        if hasattr(self.tokenizer, "encode"):
            encoded = self.tokenizer.encode(prompt)
            if isinstance(encoded, list):
                prompt_tensor = torch.tensor([encoded], dtype=torch.long, device=self.device)
            elif isinstance(encoded, torch.Tensor):
                prompt_tensor = encoded.to(self.device)
                if prompt_tensor.ndim == 1:
                    prompt_tensor = prompt_tensor.unsqueeze(0)
            else:
                prompt_tensor = torch.tensor([list(encoded)], dtype=torch.long, device=self.device)
        else:
            raise InferenceValidationError("Tokenizer has no encode method.")

        start_time = time.time()
        prompt_len = prompt_tensor.size(-1)

        # 2. Autoregressive Token Generation
        output_tensor = self.generate_tokens(prompt_tensor, config=cfg)
        elapsed = time.time() - start_time

        generated_len = output_tensor.size(-1) - prompt_len

        # 3. Decode Output Tokens to Text
        flat_output = output_tensor[0].tolist()
        if hasattr(self.tokenizer, "decode"):
            full_text = self.tokenizer.decode(flat_output)
        else:
            full_text = str(flat_output)

        stats = InferenceStatistics.compute_stats(prompt_len, generated_len, elapsed)
        logger.info(
            "Generated %d tokens in %.2fs (%.2f tokens/sec).",
            stats.generated_token_count,
            stats.duration_seconds,
            stats.tokens_per_second,
        )

        return full_text

    def generate_tokens(
        self, prompt_tokens: torch.Tensor, config: Optional[InferenceConfig] = None
    ) -> torch.Tensor:
        """Autoregressively generates new token IDs.

        Args:
            prompt_tokens: Tensor of shape (batch_size, prompt_len).
            config: Optional InferenceConfig override.

        Returns:
            Tensor of shape (batch_size, prompt_len + generated_len).
        """
        cfg = config or self.config
        self.model.eval()

        curr_tokens = prompt_tokens.to(self.device)
        batch_size = curr_tokens.size(0)

        with torch.no_grad():
            for _ in range(cfg.max_new_tokens):
                # Context Truncation to max sequence length if needed
                max_ctx = getattr(self.model, "config", None)
                max_len = getattr(max_ctx, "max_seq_len", 2048) if max_ctx else 2048
                ctx_tokens = InferenceUtilities.truncate_prompt_tokens(curr_tokens, max_allowed_tokens=max_len)

                # Forward Pass
                logits = self.model(ctx_tokens)

                # Extract last position logits (batch_size, vocab_size)
                if logits.ndim == 3:
                    next_token_logits = logits[:, -1, :]
                else:
                    next_token_logits = logits

                # Repetition Penalty
                if cfg.repetition_penalty > 1.0:
                    next_token_logits = InferenceUtilities.apply_repetition_penalty(
                        next_token_logits, curr_tokens, penalty=cfg.repetition_penalty
                    )

                # Sample Next Token
                next_token = self.composite_strategy.sample_next_token(next_token_logits, cfg)

                # Append Next Token
                curr_tokens = torch.cat([curr_tokens, next_token], dim=-1)

                # Stop if all sequences hit EOS token ID
                if (next_token == cfg.eos_token_id).all():
                    break

        return curr_tokens

    def generate_stream(
        self, prompt: str, max_new_tokens: Optional[int] = None
    ) -> Generator[str, None, None]:
        """Yields generated token text chunks as a streaming generator.

        Args:
            prompt: Input text prompt string.
            max_new_tokens: Optional max tokens override.

        Yields:
            Decoded string text token chunks.
        """
        cfg = self._override_config(max_new_tokens=max_new_tokens)
        encoded = self.tokenizer.encode(prompt)
        prompt_tensor = torch.tensor([encoded], dtype=torch.long, device=self.device)

        curr_tokens = prompt_tensor
        self.model.eval()

        with torch.no_grad():
            for _ in range(cfg.max_new_tokens):
                logits = self.model(curr_tokens)
                next_logits = logits[:, -1, :] if logits.ndim == 3 else logits
                next_token = self.composite_strategy.sample_next_token(next_logits, cfg)
                curr_tokens = torch.cat([curr_tokens, next_token], dim=-1)

                token_id = next_token[0, 0].item()
                chunk_text = self.tokenizer.decode([token_id])
                yield chunk_text

                if token_id == cfg.eos_token_id:
                    break

    def _override_config(
        self,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
    ) -> InferenceConfig:
        """Helper creating updated InferenceConfig object with overrides."""
        return InferenceConfig(
            max_new_tokens=max_new_tokens if max_new_tokens is not None else self.config.max_new_tokens,
            temperature=temperature if temperature is not None else self.config.temperature,
            top_k=top_k if top_k is not None else self.config.top_k,
            top_p=top_p if top_p is not None else self.config.top_p,
            do_sample=do_sample if do_sample is not None else self.config.do_sample,
            repetition_penalty=self.config.repetition_penalty,
            eos_token_id=self.config.eos_token_id,
            pad_token_id=self.config.pad_token_id,
        )
