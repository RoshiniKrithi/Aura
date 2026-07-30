"""Evaluation Module.

WHY THIS MODULE EXISTS:
    Evaluates model capabilities across standard code generation benchmarks (HumanEval, MBPP)
    and custom DSA problem-solving benchmarks. Includes sandboxed Python execution for code correctness tests
    and Perplexity evaluation metrics.

HOW FUTURE MODULES WILL PLUG IN:
    - Phase 17: Will implement `CodeExecutionSandbox`, `PassAtKMetric`, and `PerplexityEvaluator`.
    - Integrates with `src/inference/` to generate candidate solutions and test against automated unit test suites.
"""
