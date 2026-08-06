"""CLI Launcher Script for Experiment EXP-003 Pre-Training.

Usage:
    python scripts/run_exp_003_pretraining.py --max-steps 100 --save-interval 50 --eval-interval 50
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.exp_003_orchestrator import (
    ProgrammingPretrainingConfig,
    ProgrammingPretrainingRunner,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Aura EXP-003 Programming Pre-Training")
    parser.add_argument("--max-steps", type=int, default=50, help="Maximum pre-training steps")
    parser.add_argument("--eval-interval", type=int, default=25, help="Validation evaluation step interval")
    parser.add_argument("--save-interval", type=int, default=50, help="Checkpoint saving step interval")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt file to resume")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = ProgrammingPretrainingConfig(
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        micro_batch_size=4,
        gradient_accumulation_steps=1,
        max_sequence_length=128,
        d_model=128,
        n_layers=2,
        n_heads=2,
        d_ff=256,
        warmup_steps=0,
    )

    logger.info("Initializing ProgrammingPretrainingRunner for EXP-003...")
    runner = ProgrammingPretrainingRunner(config=config, resume_from_checkpoint=args.resume)
    summary = runner.run_pretraining()

    logger.info("==================================================")
    logger.info("EXP-003 PRE-TRAINING FINISHED")
    logger.info("Summary: %s", summary)
    logger.info("==================================================")


if __name__ == "__main__":
    main()
