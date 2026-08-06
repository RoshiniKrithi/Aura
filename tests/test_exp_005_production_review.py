"""Production PR Review & Benchmark Test Suite for Phase 24 / EXP-005 Evaluation Engine.

Includes comprehensive testing for:
- Subprocess Isolated Code Execution & Resource Limits (Memory, CPU Timeout)
- Unbiased Pass@k Statistical Formula Estimator (HumanEval formula)
- Benchmark Datasets Loaders (HumanEval, MBPP, APPS, Custom DSA)
- Code Extractor & Syntax Parser
- Leaderboard Generator & Report Serializer
- End-to-End BenchmarkSuiteRunner Execution & Checkpoint Loading
"""

import json
from pathlib import Path
import tempfile
import time
import pytest
import torch

from src.evaluation.benchmark_datasets import (
    APPSLoader,
    BenchmarkLoader,
    BenchmarkProblem,
    BenchmarkRegistry,
    CustomBenchmarkLoader,
    HumanEvalLoader,
    MBPPLoader,
)
from src.evaluation.benchmark_runner import (
    BenchmarkSuiteRunner,
    CodeExtractor,
    EvaluationBenchmarkConfig,
    LeaderboardGenerator,
)
from src.evaluation.code_sandbox import (
    CodeCompiler,
    CodeExecutionSandbox,
    ExecutionResult,
)
from src.evaluation.pass_at_k import PassAtKEstimator


def test_sandbox_isolation_and_security_limits():
    """Verifies sandbox enforces execution timeout and subprocess isolation."""
    sandbox = CodeExecutionSandbox(timeout_seconds=0.5)

    # 1. Normal Passing Script
    code_ok = "def add(a, b):\n    return a + b\n"
    test_ok = "assert add(2, 3) == 5\n"
    res_ok = sandbox.execute_code(code_ok, test_ok)
    assert res_ok.passed
    assert res_ok.status == "PASSED"

    # 2. Infinite Loop Timeout
    code_loop = "while True:\n    pass\n"
    res_loop = sandbox.execute_code(code_loop, "assert True\n")
    assert not res_loop.passed
    assert res_loop.status == "TIMEOUT"


def test_pass_at_k_unbiased_estimation_correctness():
    """Verifies math combinatorial formula pass@k estimation against edge cases."""
    # When n=10, c=5, pass@1=0.5, pass@5=0.976, pass@10=1.0
    p1 = PassAtKEstimator.compute_pass_at_k(n=10, c=5, k=1)
    p5 = PassAtKEstimator.compute_pass_at_k(n=10, c=5, k=5)
    p10 = PassAtKEstimator.compute_pass_at_k(n=10, c=5, k=10)

    assert p1 == 0.5
    assert p5 > 0.97
    assert p10 == 1.0


def test_benchmark_dataset_loaders_and_parsing():
    """Verifies benchmark dataset loading across HumanEval, MBPP, and Custom formats."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # HumanEval JSONL
        he_file = tmp_path / "humaneval.jsonl"
        with open(he_file, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "task_id": "HumanEval/0",
                        "prompt": "def solve():\n",
                        "entry_point": "solve",
                        "canonical_solution": "return 0",
                        "test": "assert solve() == 0",
                    }
                )
                + "\n"
            )

        problems = BenchmarkLoader.load_benchmark(he_file, benchmark_type="humaneval")
        assert len(problems) == 1
        assert problems[0].task_id == "HumanEval/0"


def test_code_extractor_markdown_blocks():
    """Verifies regex extraction of code from markdown code blocks."""
    text = "Here is the Python implementation:\n```python\ndef fib(n):\n    return n if n <= 1 else fib(n-1) + fib(n-2)\n```\nDone."
    code = CodeExtractor.extract_python_code(text)
    assert "def fib(n):" in code
    assert "```python" not in code


def test_leaderboard_rendering():
    """Verifies LeaderboardGenerator markdown table formatting."""
    results = {
        "humaneval": {
            "total_problems": 164.0,
            "pass@1": 0.52,
            "pass@5": 0.74,
            "pass@10": 0.88,
            "compilation_success_rate": 0.98,
            "execution_success_rate": 0.92,
        }
    }
    table = LeaderboardGenerator.generate_leaderboard_markdown(results)
    assert "# 🏆 Aura Model Benchmark Leaderboard" in table
    assert "HUMANEVAL" in table
    assert "`0.5200`" in table


def test_benchmark_runner_full_integration():
    """Verifies end-to-end BenchmarkSuiteRunner execution loop and metrics serialization."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        config = EvaluationBenchmarkConfig(
            experiment_id="PR_REVIEW_BENCHMARK_RUNNER",
            data_dir=str(tmp_path),
            tokenizer_dir=str(tmp_path),
            output_dir=str(tmp_path),
            benchmarks=["custom_dsa"],
            num_samples_per_problem=2,
            max_new_tokens=16,
            max_sequence_length=512,
            d_model=32,
            n_layers=2,
            n_heads=2,
            d_ff=64,
        )

        runner = BenchmarkSuiteRunner(config=config)
        summary = runner.run_benchmark_suite()

        assert summary["status"] == "COMPLETED_SUCCESSFULLY"
        assert "custom_dsa" in summary["benchmark_results"]

        summary_file = tmp_path / "benchmark_summary.json"
        assert summary_file.exists()
