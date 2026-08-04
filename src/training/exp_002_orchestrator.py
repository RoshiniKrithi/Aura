"""Orchestrator Engine for Phase 21 / Experiment EXP-002 Pipeline.

Coordinates multi-language code discovery, AST cleaning, BPE subword tokenizer training,
parallel binary memmap sharding, and zero-copy DataLoader benchmarking.
"""

import json
import logging
from pathlib import Path
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader

from src.datasets.binary_writer import BinaryDatasetWriter
from src.datasets.code_cleaner import CodeTextCleaner
from src.datasets.memmap_dataset import MemmapCodeDataset
from src.tokenizer.bpe_trainer import BPETrainer
from src.tokenizer.bpe_vocab import BPESpecialTokens, BPEVocab
from src.tokenizer.code_bpe_tokenizer import CodeBPESpecialTokens, CodeBPETokenizer

logger = logging.getLogger(__name__)


@dataclass
class EXP002Config:
    """Production configuration schema for Experiment EXP-002 pipeline.

    Attributes:
        experiment_id: Unique string identifier for experiment tracking.
        phase: Project roadmap phase tag.
        seed: Random seed for deterministic reproducibility.
        raw_data_dir: Input raw code datasets folder.
        output_cache_dir: Output folder for processed binary shards.
        tokenizer_dir: Output folder for BPE vocabulary artifacts.
        vocab_size: Target subword vocabulary size (default: 50257).
        max_shard_size_mb: Maximum size per binary shard file in megabytes.
        dtype: Binary shard tensor data type ("uint16" or "uint32").
        sequence_length: Sliding window sequence length L (default: 1024).
        stride: Sequence window stride step size S (default: 1024).
        train_ratio: Training split ratio (default: 0.9).
        val_ratio: Validation split ratio (default: 0.05).
        test_ratio: Testing split ratio (default: 0.05).
    """

    experiment_id: str = "EXP-002_BPE_CodePipeline_v1.0"
    phase: str = "Phase 21"
    seed: int = 42
    raw_data_dir: str = "data/raw"
    output_cache_dir: str = "data/cache/exp_002_bpe"
    tokenizer_dir: str = "data/tokenizer"

    vocab_size: int = 50257
    max_shard_size_mb: int = 1024
    dtype: str = "uint16"

    sequence_length: int = 1024
    stride: int = 1024

    train_ratio: float = 0.9
    val_ratio: float = 0.05
    test_ratio: float = 0.05

    def to_dict(self) -> Dict[str, Any]:
        """Converts EXP002Config dataclass to dictionary."""
        return asdict(self)


class PipelineOrchestrator:
    """High-performance pipeline orchestrator executing Experiment EXP-002."""

    def __init__(self, config: Optional[EXP002Config] = None) -> None:
        """Initializes PipelineOrchestrator.

        Args:
            config: Optional EXP002Config instance.
        """
        self.config = config or EXP002Config()
        self.cleaner = CodeTextCleaner(
            remove_comments=False,
            preserve_docstrings=True,
            normalize_indentation=True,
            strict_ast_validation=False,  # Permissive for multi-language code snippets
        )
        self.output_cache = Path(self.config.output_cache_dir).resolve()
        self.tokenizer_dir = Path(self.config.tokenizer_dir).resolve()
        self.output_cache.mkdir(parents=True, exist_ok=True)
        self.tokenizer_dir.mkdir(parents=True, exist_ok=True)

    def discover_code_files(self) -> List[Path]:
        """Discovers raw code source files (.py, .cpp, .hpp, .java, .txt, .jsonl)."""
        raw_path = Path(self.config.raw_data_dir).resolve()
        if not raw_path.exists():
            raw_path.mkdir(parents=True, exist_ok=True)
            # Create synthetic default code sample for initial runs if empty
            sample_code_path = raw_path / "sample_dsa.py"
            sample_code_path.write_text(
                "def binary_search(arr: list, target: int) -> int:\n"
                "    low, high = 0, len(arr) - 1\n"
                "    while low <= high:\n"
                "        mid = (low + high) // 2\n"
                "        if arr[mid] == target:\n"
                "            return mid\n"
                "        elif arr[mid] < target:\n"
                "            low = mid + 1\n"
                "        else:\n"
                "            high = mid - 1\n"
                "    return -1\n"
            )

        valid_exts = {".py", ".cpp", ".c", ".h", ".hpp", ".java", ".txt", ".jsonl"}
        found_files: List[Path] = []
        for p in raw_path.rglob("*"):
            if p.is_file() and p.suffix.lower() in valid_exts:
                found_files.append(p)

        logger.info("Discovered %d source code files in %s.", len(found_files), raw_path)
        return found_files

    def train_bpe_tokenizer(self, code_files: List[Path]) -> CodeBPETokenizer:
        """Trains or loads BPE tokenizer vocabulary and merges files.

        Args:
            code_files: List of discovered code file paths.

        Returns:
            Trained CodeBPETokenizer instance.
        """
        vocab_file = self.tokenizer_dir / "bpe_vocab_50257.json"
        merges_file = self.tokenizer_dir / "bpe_merges_50257.txt"

        if vocab_file.exists() and merges_file.exists():
            logger.info("Loading pre-trained BPE tokenizer from %s...", self.tokenizer_dir)
            return CodeBPETokenizer.from_files(vocab_file, merges_file)

        logger.info("Training new subword BPE tokenizer (Target Vocab Size: %d)...", self.config.vocab_size)
        corpus_texts: List[str] = []
        for p in code_files:
            cleaned_text, _ = self.cleaner.process_file(p)
            if cleaned_text:
                corpus_texts.append(cleaned_text)

        special_toks = CodeBPESpecialTokens()
        trainer = BPETrainer(
            vocab_size=self.config.vocab_size,
            special_tokens=special_toks,
        )
        vocab = trainer.train(corpus_texts)
        vocab.save(vocab_file, merges_file)

        return CodeBPETokenizer(vocab)

    def process_and_shard_data(
        self, code_files: List[Path], tokenizer: CodeBPETokenizer
    ) -> Tuple[List[Path], List[Path]]:
        """Cleans, tokenizes, and shards code corpus into train and val binary files.

        Args:
            code_files: List of discovered code file paths.
            tokenizer: CodeBPETokenizer instance.

        Returns:
            Tuple of (train_shard_paths, val_shard_paths).
        """
        train_writer = BinaryDatasetWriter(
            output_dir=self.output_cache,
            shard_prefix="train",
            max_shard_size_bytes=self.config.max_shard_size_mb * 1024 * 1024,
            vocab_size=self.config.vocab_size,
            dtype=self.config.dtype,
        )
        val_writer = BinaryDatasetWriter(
            output_dir=self.output_cache,
            shard_prefix="val",
            max_shard_size_bytes=self.config.max_shard_size_mb * 1024 * 1024,
            vocab_size=self.config.vocab_size,
            dtype=self.config.dtype,
        )

        n_files = len(code_files)
        val_cutoff = int(n_files * (1.0 - self.config.val_ratio)) if n_files > 1 else 1

        for i, path in enumerate(code_files):
            cleaned_text, _ = self.cleaner.process_file(path)
            if not cleaned_text:
                continue

            token_ids = tokenizer.encode_code(cleaned_text, add_special_tokens=True)

            if n_files == 1:
                train_writer.write_tokens(token_ids)
                val_writer.write_tokens(token_ids)
            elif i >= val_cutoff:
                val_writer.write_tokens(token_ids)
            else:
                train_writer.write_tokens(token_ids)

        train_summary = train_writer.close()
        val_summary = val_writer.close()

        train_paths = [Path(s["path"]) for s in train_summary["shards"]]
        val_paths = [Path(s["path"]) for s in val_summary["shards"]]

        return train_paths, val_paths

    def execute_pipeline(self) -> Dict[str, Any]:
        """Executes the end-to-end Phase 21 / EXP-002 pipeline execution lifecycle.

        Returns:
            Summary dictionary containing performance and throughput metrics.
        """
        start_time = time.time()
        logger.info("==================================================")
        logger.info("STARTING EXPERIMENT PIPELINE: %s", self.config.experiment_id)
        logger.info("==================================================")

        # 1. Discover Code Files
        code_files = self.discover_code_files()

        # 2. Train or Load BPE Tokenizer
        tokenizer = self.train_bpe_tokenizer(code_files)

        # 3. Clean, Tokenize, and Shard Data into Binary Files
        train_shards, val_shards = self.process_and_shard_data(code_files, tokenizer)

        # 4. Verify Zero-Copy Memmap DataLoader Ingestion
        dataset = MemmapCodeDataset(
            shard_paths=train_shards,
            sequence_length=self.config.sequence_length,
            stride=self.config.stride,
            dtype=self.config.dtype,
            name="TrainMemmapCodeDS",
        )

        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        first_batch_shape = None
        if len(dataset) > 0:
            for x, y in loader:
                first_batch_shape = (list(x.shape), list(y.shape))
                break
        dataset.close()

        elapsed = time.time() - start_time
        summary = {
            "experiment_id": self.config.experiment_id,
            "status": "COMPLETED_SUCCESSFULLY",
            "elapsed_seconds": round(elapsed, 2),
            "files_processed": len(code_files),
            "vocab_size": tokenizer.vocab_size,
            "train_shards_count": len(train_shards),
            "val_shards_count": len(val_shards),
            "total_sequences_indexed": len(dataset),
            "first_batch_tensor_shapes": first_batch_shape,
        }

        # Save pipeline execution summary artifact
        summary_path = self.output_cache / "pipeline_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info("==================================================")
        logger.info("EXP-002 PIPELINE COMPLETED IN %.2fs", elapsed)
        logger.info("Summary report saved to %s", summary_path)
        logger.info("==================================================")

        return summary
