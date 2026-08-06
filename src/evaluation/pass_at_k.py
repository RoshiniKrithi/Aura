"""Unbiased Pass@k Statistical Estimator for Aura EXP-005 Evaluation Suite.

Provides mathematical calculation of pass@1, pass@5, and pass@10 metrics based on the
unbiased estimator formula introduced by Chen et al. (HumanEval).
"""

import math
from typing import Dict, List, Tuple
import numpy as np


class PassAtKEstimator:
    """Computes unbiased statistical pass@k metrics for code generation evaluation."""

    @staticmethod
    def compute_pass_at_k(n: int, c: int, k: int) -> float:
        """Calculates unbiased pass@k for a single problem instance.

        Formula:
            pass@k = 1 - comb(n - c, k) / comb(n, k)

        Args:
            n: Total number of generated code candidate samples (n >= k).
            c: Number of correct candidate samples passing all unit tests.
            k: Evaluation rank sample count (e.g. 1, 5, 10).

        Returns:
            Float pass@k probability in range [0.0, 1.0].
        """
        if n < k:
            raise ValueError(f"Sample count n ({n}) must be greater than or equal to k ({k}).")

        if c < 0 or c > n:
            raise ValueError(f"Correct sample count c ({c}) must satisfy 0 <= c <= n ({n}).")

        if n - c < k:
            return 1.0

        try:
            val = 1.0 - (math.comb(n - c, k) / math.comb(n, k))
            return max(0.0, min(1.0, float(val)))
        except (ValueError, ZeroDivisionError):
            return 0.0

    @classmethod
    def compute_dataset_pass_at_k(
        cls,
        results: List[Tuple[int, int]],
        k_values: List[int] = [1, 5, 10],
    ) -> Dict[str, float]:
        """Calculates mean pass@k metrics across an entire evaluation dataset.

        Args:
            results: List of (n, c) tuples where n is total samples and c is correct count.
            k_values: List of k values to calculate (e.g. [1, 5, 10]).

        Returns:
            Dictionary mapping metric string (e.g. "pass@1") to mean float probability.
        """
        if not results:
            return {f"pass@{k}": 0.0 for k in k_values}

        metrics: Dict[str, float] = {}

        for k in k_values:
            pass_k_list = []
            for n, c in results:
                if n >= k:
                    pass_k_list.append(cls.compute_pass_at_k(n, c, k))

            if pass_k_list:
                metrics[f"pass@{k}"] = round(float(np.mean(pass_k_list)), 4)
            else:
                metrics[f"pass@{k}"] = 0.0

        return metrics
