#!/usr/bin/env python3
"""Launcher script for Phase 21: Experiment EXP-002 (BPE Subword & Binary Memmap Data Pipeline).

Executes code corpus discovery, AST cleaning, BPE subword tokenizer training,
parallel binary shard writing (uint16), and zero-copy MemmapCodeDataset DataLoader verification.
"""

import argparse
import logging
import sys

from src.training.exp_002_orchestrator import EXP002Config, PipelineOrchestrator


def setup_logging() -> None:
    """Configures structured logging for console output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main() -> None:
    """Parses command-line arguments and launches EXP-002 pipeline."""
    setup_logging()

    parser = argparse.ArgumentParser(description="Run Aura EXP-002 Data Pipeline")
    parser.add_argument("--raw-data-dir", type=str, default="data/raw", help="Raw input code folder")
    parser.add_argument("--output-cache-dir", type=str, default="data/cache/exp_002_bpe", help="Processed binary output cache folder")
    parser.add_argument("--tokenizer-dir", type=str, default="data/tokenizer", help="Tokenizer output folder")
    parser.add_argument("--vocab-size", type=int, default=50257, help="Target vocabulary size (default: 50257)")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length window (default: 1024)")

    args = parser.parse_args()

    config = EXP002Config(
        raw_data_dir=args.raw_data_dir,
        output_cache_dir=args.output_cache_dir,
        tokenizer_dir=args.tokenizer_dir,
        vocab_size=args.vocab_size,
        sequence_length=args.seq_len,
    )

    orchestrator = PipelineOrchestrator(config=config)
    summary = orchestrator.execute_pipeline()

    print("\n" + "=" * 60)
    print("EXP-002 PIPELINE RUN COMPLETE")
    print(f"Summary Metrics: {summary}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
