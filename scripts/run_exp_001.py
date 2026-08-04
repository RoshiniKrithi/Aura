#!/usr/bin/env python3
"""Launcher script for Phase 20: Experiment EXP-001 (Tiny Shakespeare Baseline Training).

Executes end-to-end training lifecycle, logging, checkpointing, sampling (every 100 iterations),
and artifact generation for Aura LLM.
"""

import argparse
import logging
import sys

from src.training.exp_001 import ExperimentConfig, ExperimentRunner


def setup_logging() -> None:
    """Configures structured logging for console output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main() -> None:
    """Parses command-line arguments and launches EXP-001."""
    setup_logging()

    parser = argparse.ArgumentParser(description="Run Aura EXP-001 Tiny Shakespeare Experiment")
    parser.add_argument("--max-steps", type=int, default=300, help="Maximum training steps (default: 300)")
    parser.add_argument("--device", type=str, default="auto", help="Execution device (auto, cpu, cuda)")
    parser.add_argument("--sample-interval", type=int, default=100, help="Text sample generation step interval (default: 100)")
    parser.add_argument("--output-dir", type=str, default="outputs/experiments/EXP-001_TinyShakespeare_v1.0", help="Output directory")

    args = parser.parse_args()

    config = ExperimentConfig(
        max_steps=args.max_steps,
        device=args.device,
        sample_interval=args.sample_interval,
        output_dir=args.output_dir,
    )

    runner = ExperimentRunner(config=config)
    summary = runner.run_experiment()

    print("\n" + "=" * 60)
    print("EXP-001 RUN COMPLETE")
    print(f"Summary Metrics: {summary}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
