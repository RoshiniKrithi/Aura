#!/usr/bin/env python3
"""Launcher script for Phase 20: Experiment EXP-001 (Tiny Shakespeare Baseline Training).

Executes end-to-end training lifecycle, logging, checkpointing, sampling (every 100 iterations),
and artifact generation for Aura LLM.
"""

import argparse
import logging
from pathlib import Path
import sys

# Ensure workspace root is in sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum training steps (default: 1000)")
    parser.add_argument("--d-model", type=int, default=128, help="Embedding dimension (default: 128)")
    parser.add_argument("--n-layers", type=int, default=4, help="Transformer layers (default: 4)")
    parser.add_argument("--n-heads", type=int, default=4, help="Attention heads (default: 4)")
    parser.add_argument("--learning-rate", type=float, default=3.0e-4, help="Peak learning rate (default: 3e-4)")
    parser.add_argument("--warmup-steps", type=int, default=500, help="Warmup steps (default: 500)")
    parser.add_argument("--batch-size", type=int, default=32, help="Global batch size (default: 32)")
    parser.add_argument("--device", type=str, default="auto", help="Execution device (auto, cpu, cuda)")
    parser.add_argument("--sample-interval", type=int, default=500, help="Text sample generation step interval (default: 500)")
    parser.add_argument("--output-dir", type=str, default="outputs/experiments/EXP-001_TinyShakespeare_v1.0", help="Output directory")

    args = parser.parse_args()

    config = ExperimentConfig(
        max_steps=args.max_steps,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        global_batch_size=args.batch_size,
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
