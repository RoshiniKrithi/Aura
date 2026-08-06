"""Performance Optimization Configuration Dataclasses for Aura EXP-008.

Provides PerformanceConfig and QuantizationType enum.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class QuantizationType(str, Enum):
    """Supported model quantization precision modes."""

    NONE = "none"
    INT8_DYNAMIC = "int8_dynamic"
    INT8_STATIC = "int8_static"
    INT4_WEIGHT_ONLY = "int4_weight_only"


@dataclass
class PerformanceConfig:
    """Configuration container for performance optimization hyperparameters.

    Attributes:
        experiment_id: Unique string identifier for experiment tracking.
        use_flash_attention: Enables FlashAttention-2 fused kernel execution.
        use_kv_cache: Enables pre-allocated continuous Paged KV-Cache.
        use_amp: Enables Automatic Mixed Precision (AMP).
        amp_dtype: Precision string for AMP ("float16" or "bfloat16").
        use_compile: Enables torch.compile JIT graph optimization.
        compile_mode: JIT compilation mode ("default", "reduce-overhead", "max-autotune").
        quantization_type: Active quantization mode enum.
        max_batch_size: Maximum batch size capacity for KV cache allocation.
        max_context_length: Maximum sequence context length L.
        num_threads: CPU thread count override.
        device: Target execution device ("cuda", "cpu", "auto").
        benchmark_iterations: Number of benchmark warmup and timing iterations.
        output_dir: Output path for benchmark logs and performance reports.
    """

    experiment_id: str = "EXP-008_Performance_v1.0"
    use_flash_attention: bool = True
    use_kv_cache: bool = True
    use_amp: bool = True
    amp_dtype: str = "bfloat16"

    use_compile: bool = False
    compile_mode: str = "reduce-overhead"

    quantization_type: QuantizationType = QuantizationType.NONE

    max_batch_size: int = 4
    max_context_length: int = 2048
    num_threads: int = 4
    device: str = "auto"

    benchmark_iterations: int = 10
    output_dir: str = "outputs/experiments/EXP-008_Optimization_v1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Converts PerformanceConfig to dictionary representation."""
        res = asdict(self)
        res["quantization_type"] = str(self.quantization_type)
        return res
