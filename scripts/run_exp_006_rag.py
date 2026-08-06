"""CLI Launcher Script for Experiment EXP-006 Retrieval-Augmented Generation (RAG).

Usage:
    python scripts/run_exp_006_rag.py --query "How to implement a Min-Heap in Python?"
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.rag_orchestrator import RAGConfig, RAGOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Aura EXP-006 RAG Query")
    parser.add_argument("--query", type=str, default="Explain how to implement a Min-Heap in Python", help="User question query")
    parser.add_argument("--knowledge-dir", type=str, default="data/knowledge_base", help="Knowledge base directory")
    parser.add_argument("--model-checkpoint", type=str, default=None, help="Path to model weights checkpoint (.pt)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = RAGConfig(
        model_checkpoint_path=args.model_checkpoint,
        knowledge_dir=args.knowledge_dir,
        chunk_size=256,
        chunk_overlap=32,
        top_k_retrieval=5,
        top_k_rerank=3,
        max_sequence_length=512,
        d_model=128,
        n_layers=2,
        n_heads=2,
        d_ff=256,
    )

    logger.info("Initializing RAGOrchestrator for EXP-006...")
    orchestrator = RAGOrchestrator(config=config)
    
    logger.info("Indexing Knowledge Base...")
    chunk_count = orchestrator.build_knowledge_index()
    logger.info("Indexed %d chunks.", chunk_count)

    logger.info("Executing RAG Query: '%s'", args.query)
    res = orchestrator.query(args.query)

    logger.info("==================================================")
    logger.info("RAG GROUNDED RESPONSE:")
    logger.info(res["answer"])
    logger.info("--------------------------------------------------")
    logger.info("CITATIONS: %s", res["citations"])
    logger.info("==================================================")


if __name__ == "__main__":
    main()
