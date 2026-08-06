"""CLI Launcher Script for Experiment EXP-005 Programming Evaluation & Benchmark Suite.

Usage:
    python scripts/run_exp_005_eval.py --num-samples 10 --benchmarks humaneval mbpp custom_dsa
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.benchmark_runner import (
    BenchmarkSuiteRunner,
    EvaluationBenchmarkConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Aura EXP-005 Evaluation Benchmark Suite")
    parser.add_argument("--num-samples", type=int, default=10, help="Candidate solutions generated per problem (n=10)")
    parser.add_argument("--benchmarks", nargs="+", default=["humaneval", "mbpp", "custom_dsa"], help="Benchmark tags to run")
    parser.add_argument("--model-checkpoint", type=str, default=None, help="Path to model weights checkpoint (.pt)")
    parser.add_argument("--timeout", type=float, default=5.0, help="Sandbox subprocess execution timeout")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = EvaluationBenchmarkConfig(
        model_checkpoint_path=args.model_checkpoint,
        benchmarks=args.benchmarks,
        num_samples_per_problem=args.num_samples,
        sandbox_timeout=args.timeout,
        max_sequence_length=128,
        d_model=128,
        n_layers=2,
        n_heads=2,
        d_ff=256,
    )

    logger.info("Initializing BenchmarkSuiteRunner for EXP-005...")
    runner = BenchmarkSuiteRunner(config=config)
    summary = runner.run_benchmark_suite()

    logger.info("==================================================")
    logger.info("EXP-005 BENCHMARK EVALUATION FINISHED")
    logger.info("Summary: %s", summary)
    logger.info("==================================================")


if __name__ == "__main__":
    main()
