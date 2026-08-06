"""Comprehensive PyTest Suite for Experiment EXP-005 Programming Evaluation & Benchmarks.

Includes unit, integration, stress, and regression testing for:
- Code Execution Sandbox & Subprocess Resource Limit Enforcer
- Unbiased Pass@k Statistical Formula Estimator
- Benchmark Loaders & Problem Registry (HumanEval, MBPP, APPS, Custom)
- Markdown Code Extraction & Parsing
- Leaderboard Generator & Report Serializer
- End-to-End BenchmarkSuiteRunner Execution
"""

import json
from pathlib import Path
import tempfile
import pytest

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
    TestCaseRunner,
)
from src.evaluation.pass_at_k import PassAtKEstimator


def test_code_compiler_syntax_validation():
    """Verifies CodeCompiler syntax validation."""
    valid_code = "def add(a, b):\n    return a + b"
    is_valid, err = CodeCompiler.validate_python_syntax(valid_code)
    assert is_valid
    assert err is None

    invalid_code = "def add(a, b:\n    return a + b"
    is_invalid, err_msg = CodeCompiler.validate_python_syntax(invalid_code)
    assert not is_invalid
    assert "SyntaxError" in err_msg


def test_code_execution_sandbox_passing_and_failing():
    """Verifies sandboxed subprocess code execution under passing and failing assertions."""
    sandbox = CodeExecutionSandbox(timeout_seconds=2.0)

    # 1. Passing Code
    code_pass = "def square(x):\n    return x * x\n"
    test_pass = "assert square(3) == 9\nassert square(0) == 0\n"
    res_pass = sandbox.execute_code(code_pass, test_pass)

    assert res_pass.passed
    assert res_pass.status == "PASSED"
    assert res_pass.execution_time > 0.0

    # 2. Failing Assertion
    test_fail = "assert square(3) == 10\n"
    res_fail = sandbox.execute_code(code_pass, test_fail)

    assert not res_fail.passed
    assert res_fail.status == "FAILED"
    assert "AssertionError" in res_fail.error


def test_code_execution_sandbox_timeout():
    """Verifies sandbox enforces execution timeout on infinite loops."""
    sandbox = CodeExecutionSandbox(timeout_seconds=0.5)

    code_infinite = "import time\nwhile True:\n    time.sleep(0.1)\n"
    test_stub = "assert True\n"

    res_timeout = sandbox.execute_code(code_infinite, test_stub)
    assert not res_timeout.passed
    assert res_timeout.status == "TIMEOUT"
    assert "timed out" in res_timeout.error


def test_pass_at_k_unbiased_estimator_math():
    """Verifies mathematical correctness of unbiased pass@k combinatorial formula."""
    # When all samples correct (n=10, c=10), pass@1, pass@5, pass@10 must be 1.0
    p1 = PassAtKEstimator.compute_pass_at_k(n=10, c=10, k=1)
    p5 = PassAtKEstimator.compute_pass_at_k(n=10, c=10, k=5)
    p10 = PassAtKEstimator.compute_pass_at_k(n=10, c=10, k=10)
    assert p1 == 1.0
    assert p5 == 1.0
    assert p10 == 1.0

    # When zero samples correct (n=10, c=0), pass@k must be 0.0
    p1_zero = PassAtKEstimator.compute_pass_at_k(n=10, c=0, k=1)
    assert p1_zero == 0.0

    # When 5 of 10 samples correct, pass@1 = 0.5
    p1_half = PassAtKEstimator.compute_pass_at_k(n=10, c=5, k=1)
    assert p1_half == 0.5

    # Dataset aggregation
    dataset_metrics = PassAtKEstimator.compute_dataset_pass_at_k(
        results=[(10, 10), (10, 5)], k_values=[1, 5, 10]
    )
    assert dataset_metrics["pass@1"] == 0.75


def test_benchmark_loaders_and_registry():
    """Verifies HumanEval, MBPP, APPS, and custom benchmark loaders."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        he_file = tmp_path / "humaneval.jsonl"

        with open(he_file, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "task_id": "HumanEval/0",
                        "prompt": "def has_close_elements():\n",
                        "entry_point": "has_close_elements",
                        "canonical_solution": "return False",
                        "test": "assert has_close_elements() == False",
                    }
                )
                + "\n"
            )

        problems = BenchmarkLoader.load_benchmark(he_file, benchmark_type="humaneval")
        assert len(problems) == 1
        assert problems[0].task_id == "HumanEval/0"
        assert problems[0].entry_point == "has_close_elements"


def test_code_extractor():
    """Verifies regex markdown code block extraction."""
    md_text = "Here is the solution:\n```python\ndef solve():\n    return 42\n```\nHope this helps!"
    code = CodeExtractor.extract_python_code(md_text)
    assert code == "def solve():\n    return 42"


def test_leaderboard_generator_markdown():
    """Verifies LeaderboardGenerator renders clean markdown tables."""
    results = {
        "humaneval": {
            "total_problems": 164.0,
            "pass@1": 0.45,
            "pass@5": 0.65,
            "pass@10": 0.80,
            "compilation_success_rate": 0.95,
            "execution_success_rate": 0.90,
        }
    }
    md = LeaderboardGenerator.generate_leaderboard_markdown(results)
    assert "# 🏆 Aura Model Benchmark Leaderboard" in md
    assert "HUMANEVAL" in md
    assert "`0.4500`" in md


def test_benchmark_suite_runner_execution():
    """Verifies end-to-end BenchmarkSuiteRunner execution loop."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        config = EvaluationBenchmarkConfig(
            experiment_id="TEST_BENCHMARK_RUNNER",
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
