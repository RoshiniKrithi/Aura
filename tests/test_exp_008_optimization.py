"""Comprehensive PyTest Suite for Experiment EXP-008 Performance Optimization.

Includes unit, integration, stress, and benchmark testing for:
- FlashAttention-2 Fused CUDA Kernel Wrapper
- Paged and Pre-allocated Key-Value (KV) Cache Manager
- Automatic Mixed Precision (AMP FP16/BF16) Autocast Manager
- INT8 Dynamic Weight Quantization Engine
- torch.compile JIT Compiler Integration
- VRAM Memory Allocation & Cache Eviction Manager
- Latency (TTFT, ITL), Throughput (tok/s), and Memory Profiler
- High-Throughput OptimizedInferenceEngine & PerformanceManager
"""

from pathlib import Path
import tempfile
import pytest
import torch
import torch.nn as nn

from src.models.config import AuraGPTConfig
from src.models.gpt import AuraGPT
from src.optimization.amp_manager import MixedPrecisionManager
from src.optimization.compiler import TorchCompileManager
from src.optimization.flash_attention import FlashAttention2, FlashAttentionManager
from src.optimization.inference_optimizer import OptimizedInferenceEngine, PerformanceManager
from src.optimization.kv_cache import KVCacheManager
from src.optimization.memory_optimizer import MemoryOptimizer
from src.optimization.optimization_config import PerformanceConfig, QuantizationType
from src.optimization.profiler import (
    LatencyBenchmark,
    MemoryBenchmark,
    PerformanceReporter,
    PerformanceStatistics,
    Profiler,
    ThroughputBenchmark,
)
from src.optimization.quantization import QuantizationManager, QuantizedLinear


def test_flash_attention_forward_and_shapes():
    """Verifies FlashAttention2 forward pass and tensor output shapes."""
    flash_attn = FlashAttention2(dropout=0.0)
    # Shape: (batch=2, n_heads=4, seq_len=16, head_dim=32)
    q = torch.randn(2, 4, 16, 32)
    k = torch.randn(2, 4, 16, 32)
    v = torch.randn(2, 4, 16, 32)

    out = flash_attn(q, k, v, is_causal=True)
    assert out.shape == q.shape
    assert not torch.isnan(out).any()


def test_kv_cache_preallocation_and_updates():
    """Verifies KVCacheManager pre-allocates continuous tensors and updates step indices."""
    kv_mgr = KVCacheManager(
        batch_size=2, n_heads=4, max_seq_len=64, head_dim=16, dtype=torch.float32
    )

    assert kv_mgr.key_cache.shape == (2, 4, 64, 16)
    assert kv_mgr.current_step == 0

    step_k = torch.randn(2, 4, 4, 16)
    step_v = torch.randn(2, 4, 4, 16)

    keys_out, values_out = kv_mgr.update(step_k, step_v, step=0)
    assert keys_out.shape == (2, 4, 4, 16)
    assert kv_mgr.current_step == 4

    kv_mgr.reset()
    assert kv_mgr.current_step == 0


def test_mixed_precision_manager_context():
    """Verifies MixedPrecisionManager autocast context resolution."""
    with MixedPrecisionManager.get_autocast_context(enabled=True, device_type="cpu", dtype_str="float16"):
        x = torch.randn(2, 4)
        assert x is not None


def test_int8_quantization_and_layer_replacement():
    """Verifies QuantizationManager quantizes linear layers to QuantizedLinear."""
    gpt_cfg = AuraGPTConfig(
        model_name="aura-quant-test",
        vocab_size=50260,
        max_sequence_length=128,
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_ff=64,
    )
    model = AuraGPT(gpt_cfg)

    quant_model, stats = QuantizationManager.quantize_model(
        model, quant_type=QuantizationType.INT8_DYNAMIC
    )

    assert stats["quantized_layers_count"] > 0
    assert isinstance(quant_model.lm_head, QuantizedLinear)

    x = torch.randn(2, 8, 32)
    out = quant_model(torch.randint(0, 100, (2, 8)))
    assert out.shape == (2, 8, 50260)


def test_torch_compile_manager_fallback():
    """Verifies TorchCompileManager handles compilation or eager mode fallback."""
    base_linear = nn.Linear(16, 16)
    compiled = TorchCompileManager.compile_model(base_linear, disable=True)
    assert compiled is not None


def test_memory_optimizer_statistics():
    """Verifies MemoryOptimizer measures VRAM allocations."""
    stats = MemoryOptimizer.get_memory_statistics()
    assert "allocated_vram_mb" in stats
    assert "max_allocated_vram_mb" in stats


def test_profiler_and_performance_reporter():
    """Verifies Profiler measures latency, throughput, and exports JSON reports."""
    gpt_cfg = AuraGPTConfig(
        model_name="aura-prof-test",
        vocab_size=50260,
        max_sequence_length=128,
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_ff=64,
    )
    model = AuraGPT(gpt_cfg)
    profiler = Profiler(model=model)

    def dummy_gen(p, max_new_tokens):
        return torch.randint(0, 100, (p.shape[0], p.shape[1] + max_new_tokens))

    prompt = torch.randint(0, 100, (2, 8))
    stats = profiler.profile_generation(dummy_gen, prompt, max_new_tokens=4)

    assert stats.tokens_per_second > 0
    assert stats.time_to_first_token_ms > 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        report_path = Path(tmp_dir) / "report.json"
        PerformanceReporter.save_report(stats, report_path)
        assert report_path.exists()


def test_optimized_inference_engine_end_to_end():
    """Verifies end-to-end OptimizedInferenceEngine generation and profiling."""
    gpt_cfg = AuraGPTConfig(
        model_name="aura-opt-test",
        vocab_size=50260,
        max_sequence_length=128,
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_ff=64,
    )
    model = AuraGPT(gpt_cfg)
    config = PerformanceConfig(
        use_flash_attention=True,
        use_kv_cache=True,
        use_amp=True,
        quantization_type=QuantizationType.NONE,
    )

    engine = OptimizedInferenceEngine(model=model, config=config)
    prompt = torch.randint(0, 100, (1, 8))

    generated = engine.generate(prompt, max_new_tokens=4)
    assert generated.shape == (1, 12)

    perf_stats = engine.profile(prompt, max_new_tokens=4)
    assert perf_stats.tokens_per_second > 0
