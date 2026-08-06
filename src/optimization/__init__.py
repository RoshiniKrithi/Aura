"""Performance Optimization, Quantization, and Acceleration Subsystem for Aura.

Provides PerformanceConfig, QuantizationType, FlashAttention2, FlashAttentionManager,
KVCacheManager, MixedPrecisionManager, QuantizedLinear, QuantizationManager,
TorchCompileManager, MemoryOptimizer, PerformanceStatistics, LatencyBenchmark,
ThroughputBenchmark, MemoryBenchmark, Profiler, PerformanceReporter,
PerformanceManager, and OptimizedInferenceEngine.
"""

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

__all__ = [
    "PerformanceConfig",
    "QuantizationType",
    "FlashAttention2",
    "FlashAttentionManager",
    "KVCacheManager",
    "MixedPrecisionManager",
    "QuantizedLinear",
    "QuantizationManager",
    "TorchCompileManager",
    "MemoryOptimizer",
    "PerformanceStatistics",
    "LatencyBenchmark",
    "ThroughputBenchmark",
    "MemoryBenchmark",
    "Profiler",
    "PerformanceReporter",
    "PerformanceManager",
    "OptimizedInferenceEngine",
]
