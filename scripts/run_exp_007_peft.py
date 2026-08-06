"""CLI Launcher Script for Experiment EXP-007 LoRA & PEFT Fine-Tuning.

Usage:
    python scripts/run_exp_007_peft.py --max-steps 20 --learning-rate 3e-4 --rank 16
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.peft.peft_config import LoRAConfig, PEFTTrainingConfig
from src.peft.peft_trainer import PEFTRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Aura EXP-007 LoRA Fine-Tuning")
    parser.add_argument("--max-steps", type=int, default=20, help="Maximum training steps")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate for adapter params")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank dimension r")
    parser.add_argument("--alpha", type=float, default=32.0, help="LoRA scaling alpha")
    parser.add_argument("--pretrained-checkpoint", type=str, default=None, help="Pretrained base model weights (.pt)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    lora_cfg = LoRAConfig(r=args.rank, alpha=args.alpha)
    config = PEFTTrainingConfig(
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        pretrained_checkpoint_path=args.pretrained_checkpoint,
        lora_config=lora_cfg,
        max_sequence_length=256,
        d_model=128,
        n_layers=2,
        n_heads=2,
        d_ff=256,
    )

    logger.info("Initializing PEFTRunner for EXP-007...")
    runner = PEFTRunner(config=config)
    summary = runner.run_peft_training()

    logger.info("==================================================")
    logger.info("EXP-007 LORA PEFT TRAINING FINISHED")
    logger.info("Summary: %s", summary)
    logger.info("==================================================")


if __name__ == "__main__":
    main()
