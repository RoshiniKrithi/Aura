"""High-Throughput Optimized Inference Engine for Aura EXP-008.

Provides OptimizedInferenceEngine and PerformanceManager for fast code generation.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

from src.inference.engine import InferenceEngine
from src.models.gpt import AuraGPT
from src.optimization.amp_manager import MixedPrecisionManager
from src.optimization.compiler import TorchCompileManager
from src.optimization.flash_attention import FlashAttentionManager
from src.optimization.kv_cache import KVCacheManager
from src.optimization.memory_optimizer import MemoryOptimizer
from src.optimization.optimization_config import PerformanceConfig
from src.optimization.profiler import PerformanceStatistics, Profiler
from src.optimization.quantization import QuantizationManager

logger = logging.getLogger(__name__)


class PerformanceManager:
    """Master orchestrator configuring hardware acceleration, quantization, and caching."""

    def __init__(self, model: nn.Module, config: PerformanceConfig) -> None:
        """Initializes PerformanceManager.

        Args:
            model: PyTorch model instance (AuraGPT).
            config: PerformanceConfig instance.
        """
        self.config = config
        self.device = self._resolve_device(config.device)

        # 1. Quantization
        self.model, self.quant_stats = QuantizationManager.quantize_model(
            model=model, quant_type=config.quantization_type
        )
        self.model = self.model.to(self.device)

        # 2. JIT Graph Compilation
        if config.use_compile:
            self.model = TorchCompileManager.compile_model(
                self.model, mode=config.compile_mode
            )

        # 3. KV Cache Pre-Allocation
        self.kv_cache: Optional[KVCacheManager] = None
        if config.use_kv_cache:
            n_heads = getattr(model.config, "n_heads", 12) if hasattr(model, "config") else 12
            d_model = getattr(model.config, "d_model", 768) if hasattr(model, "config") else 768
            head_dim = d_model // n_heads

            self.kv_cache = KVCacheManager(
                batch_size=config.max_batch_size,
                n_heads=n_heads,
                max_seq_len=config.max_context_length,
                head_dim=head_dim,
                dtype=torch.float32,
                device=self.device,
            )

        self.profiler = Profiler(model=self.model)

    def _resolve_device(self, req: str) -> torch.device:
        if req == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


class OptimizedInferenceEngine:
    """High-throughput inference engine executing accelerated code token generation."""

    def __init__(
        self,
        model: AuraGPT,
        tokenizer: Optional[Any] = None,
        config: Optional[PerformanceConfig] = None,
    ) -> None:
        """Initializes OptimizedInferenceEngine."""
        from src.tokenizer.code_bpe_tokenizer import CodeBPETokenizer
        self.config = config or PerformanceConfig()
        self.perf_mgr = PerformanceManager(model=model, config=self.config)
        tok = tokenizer or CodeBPETokenizer.create_default()
        self.base_engine = InferenceEngine(model=self.perf_mgr.model, tokenizer=tok)

    def generate(
        self,
        prompt: Union[str, torch.Tensor],
        max_new_tokens: int = 32,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> Union[str, torch.Tensor]:
        """Executes accelerated autoregressive code generation under AMP autocast.

        Returns:
            Generated sequence string or token tensor.
        """
        device = self.perf_mgr.device

        with MixedPrecisionManager.get_autocast_context(
            enabled=self.config.use_amp,
            device_type=str(device),
            dtype_str=self.config.amp_dtype,
        ):
            if isinstance(prompt, str):
                return self.base_engine.generate(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
            else:
                from src.inference.config import InferenceConfig
                cfg = InferenceConfig(
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                prompt_tokens = prompt.to(device)
                return self.base_engine.generate_tokens(prompt_tokens=prompt_tokens, config=cfg)

    def profile(
        self, prompt_tokens: torch.Tensor, max_new_tokens: int = 20
    ) -> PerformanceStatistics:
        """Profiles generation throughput, latency, and VRAM memory usage."""
        def gen_fn(p, max_new_tokens):
            return self.generate(p, max_new_tokens=max_new_tokens)

        return self.perf_mgr.profiler.profile_generation(
            generation_fn=gen_fn,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
        )
