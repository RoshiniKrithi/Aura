"""Latency, Throughput, and VRAM Memory Benchmarking Suite for Aura EXP-008.

Provides PerformanceStatistics, LatencyBenchmark, ThroughputBenchmark,
MemoryBenchmark, Profiler, and PerformanceReporter.
"""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from src.optimization.memory_optimizer import MemoryOptimizer

logger = logging.getLogger(__name__)


@dataclass
class PerformanceStatistics:
    """Statistics container holding latency, throughput, and memory benchmark metrics."""

    time_to_first_token_ms: float
    inter_token_latency_ms: float
    tokens_per_second: float
    total_tokens_generated: int
    total_execution_time_sec: float
    allocated_vram_mb: float
    max_allocated_vram_mb: float
    is_accelerated: bool

    def to_dict(self) -> Dict[str, Any]:
        """Converts PerformanceStatistics to dictionary representation."""
        return asdict(self)


class LatencyBenchmark:
    """Measures Time-to-First-Token (TTFT) and Inter-Token Latency (ITL)."""

    @staticmethod
    def measure_latency(
        generation_fn, prompt_tokens: torch.Tensor, max_new_tokens: int = 10
    ) -> Tuple[float, float]:
        """Measures TTFT and ITL in milliseconds.

        Returns:
            Tuple of (ttft_ms, itl_ms).
        """
        start_t = time.perf_counter()
        _ = generation_fn(prompt_tokens, max_new_tokens=1)
        ttft_ms = (time.perf_counter() - start_t) * 1000.0

        if max_new_tokens > 1:
            start_gen_t = time.perf_counter()
            _ = generation_fn(prompt_tokens, max_new_tokens=max_new_tokens)
            total_gen_time_ms = (time.perf_counter() - start_gen_t) * 1000.0
            itl_ms = total_gen_time_ms / max_new_tokens
        else:
            itl_ms = ttft_ms

        return round(ttft_ms, 2), round(itl_ms, 2)


class ThroughputBenchmark:
    """Measures generation throughput in tokens per second."""

    @staticmethod
    def measure_throughput(
        generation_fn, prompt_tokens: torch.Tensor, max_new_tokens: int = 20
    ) -> Tuple[float, int, float]:
        """Measures tokens per second.

        Returns:
            Tuple of (tokens_per_second, total_tokens, execution_time_sec).
        """
        start_t = time.perf_counter()
        _ = generation_fn(prompt_tokens, max_new_tokens=max_new_tokens)
        elapsed_sec = time.perf_counter() - start_t

        num_generated = max_new_tokens * prompt_tokens.shape[0]
        tok_per_sec = num_generated / max(1e-5, elapsed_sec)

        return round(tok_per_sec, 2), num_generated, round(elapsed_sec, 4)


class MemoryBenchmark:
    """Measures peak VRAM and RAM memory allocation metrics."""

    @staticmethod
    def measure_memory_usage() -> Dict[str, float]:
        """Returns peak VRAM memory allocation statistics."""
        return MemoryOptimizer.get_memory_statistics()



class Profiler:
    """Master profiler benchmarking latency, throughput, and VRAM memory usage."""

    def __init__(self, model: nn.Module) -> None:
        """Initializes Profiler."""
        self.model = model

    def profile_generation(
        self, generation_fn, prompt_tokens: torch.Tensor, max_new_tokens: int = 20
    ) -> PerformanceStatistics:
        """Runs complete performance benchmarking pipeline."""
        MemoryOptimizer.reset_peak_memory_stats()

        ttft_ms, itl_ms = LatencyBenchmark.measure_latency(
            generation_fn, prompt_tokens, max_new_tokens
        )
        tok_per_sec, total_toks, elapsed_sec = ThroughputBenchmark.measure_throughput(
            generation_fn, prompt_tokens, max_new_tokens
        )

        mem_stats = MemoryOptimizer.get_memory_statistics()

        stats = PerformanceStatistics(
            time_to_first_token_ms=ttft_ms,
            inter_token_latency_ms=itl_ms,
            tokens_per_second=tok_per_sec,
            total_tokens_generated=total_toks,
            total_execution_time_sec=elapsed_sec,
            allocated_vram_mb=mem_stats["allocated_vram_mb"],
            max_allocated_vram_mb=mem_stats["max_allocated_vram_mb"],
            is_accelerated=True,
        )

        logger.info(
            "Benchmark Complete: Throughput=%.2f tok/s | TTFT=%.2f ms | ITL=%.2f ms | VRAM=%.2f MB",
            tok_per_sec,
            ttft_ms,
            itl_ms,
            mem_stats["max_allocated_vram_mb"],
        )
        return stats


class PerformanceReporter:
    """Exports performance statistics into JSON report files."""

    @staticmethod
    def save_report(stats: PerformanceStatistics, output_path: Union[str, Path]) -> Path:
        """Saves PerformanceStatistics object to JSON file."""
        path = Path(output_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(stats.to_dict(), f, indent=2)

        logger.info("Saved performance benchmark report to %s", path)
        return path
