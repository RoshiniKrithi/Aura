"""VRAM Memory Allocation and Cache Eviction Manager for Aura EXP-008.

Provides MemoryOptimizer for tracking peak VRAM usage and clearing CUDA cache.
"""

import logging
from typing import Dict
import torch

logger = logging.getLogger(__name__)


class MemoryOptimizer:
    """Manages VRAM allocation, CUDA cache purging, and memory utilization statistics."""

    @staticmethod
    def empty_cache() -> None:
        """Purges unused cached memory allocations in CUDA VRAM."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("Flushed unused CUDA memory allocations.")

    @staticmethod
    def get_memory_statistics() -> Dict[str, float]:
        """Measures active and peak VRAM allocation statistics.

        Returns:
            Dictionary containing VRAM stats in megabytes (MB).
        """
        if not torch.cuda.is_available():
            return {
                "allocated_vram_mb": 0.0,
                "reserved_vram_mb": 0.0,
                "max_allocated_vram_mb": 0.0,
            }

        allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)

        return {
            "allocated_vram_mb": round(allocated, 2),
            "reserved_vram_mb": round(reserved, 2),
            "max_allocated_vram_mb": round(max_allocated, 2),
        }

    @staticmethod
    def reset_peak_memory_stats() -> None:
        """Resets peak memory allocation trackers."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
