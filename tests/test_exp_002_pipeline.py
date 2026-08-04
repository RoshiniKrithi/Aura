"""Unit tests for Phase 21 / Experiment EXP-002 Pipeline components.

Verifies CodeTextCleaner, BinaryDatasetWriter, MemmapCodeDataset, CodeBPETokenizer,
and PipelineOrchestrator.
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch

from src.datasets.binary_writer import BinaryDatasetWriter
from src.datasets.code_cleaner import CodeTextCleaner
from src.datasets.memmap_dataset import MemmapCodeDataset
from src.tokenizer.bpe_trainer import BPETrainer
from src.tokenizer.code_bpe_tokenizer import CodeBPESpecialTokens, CodeBPETokenizer
from src.training.exp_002_orchestrator import EXP002Config, PipelineOrchestrator


def test_code_text_cleaner() -> None:
    """Tests CodeTextCleaner AST validation and whitespace cleaning."""
    cleaner = CodeTextCleaner(remove_comments=True, strict_ast_validation=True)

    # Valid Python code
    py_code = "# Comment\ndef foo(x):\n    return x + 1\n"
    cleaned, is_valid = cleaner.clean_python(py_code)
    assert is_valid is True
    assert "def foo(x):" in cleaned
    assert "# Comment" not in cleaned

    # Invalid Python code
    bad_code = "def foo(x\n    return x +"
    _, is_valid_bad = cleaner.clean_python(bad_code)
    assert is_valid_bad is False

    # Valid C++ code
    cpp_code = "// Comment\nint main() { return 0; }"
    cleaned_cpp, is_valid_cpp = cleaner.clean_cpp(cpp_code)
    assert is_valid_cpp is True
    assert "int main()" in cleaned_cpp


def test_binary_dataset_writer_and_memmap_dataset() -> None:
    """Tests BinaryDatasetWriter sharding and MemmapCodeDataset zero-copy loading."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        writer = BinaryDatasetWriter(
            output_dir=tmp_path,
            shard_prefix="test_train",
            max_shard_size_bytes=100,  # Small shard size for testing split
            vocab_size=1000,
            dtype="uint16",
        )

        tokens = list(range(100))
        writer.write_tokens(tokens)
        summary = writer.close()

        assert summary["total_tokens_written"] == 100
        assert summary["total_shards"] >= 1

        shards = [Path(s["path"]) for s in summary["shards"]]

        # Instantiation of MemmapCodeDataset
        ds = MemmapCodeDataset(
            shard_paths=shards,
            sequence_length=16,
            stride=16,
            dtype="uint16",
            name="TestMemmapDS",
        )

        assert len(ds) > 0
        x, y = ds[0]
        assert isinstance(x, torch.LongTensor)
        assert isinstance(y, torch.LongTensor)
        assert x.shape == (16,)
        assert y.shape == (16,)
        assert (x[1:] == y[:-1]).all()  # Check causal shift by 1
        ds.close()


def test_code_bpe_tokenizer() -> None:
    """Tests CodeBPETokenizer encoding and decoding roundtrips with specialized DSA tokens."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        corpus = [
            "def binary_search(arr, target):",
            "    return -1",
            "int main() { return 0; }",
        ]

        special_toks = CodeBPESpecialTokens()
        trainer = BPETrainer(vocab_size=200, special_tokens=special_toks)
        vocab = trainer.train(corpus)

        vocab_file = tmp_path / "vocab.json"
        merges_file = tmp_path / "merges.txt"
        vocab.save(vocab_file, merges_file)

        tok = CodeBPESpecialTokens()
        tokenizer = CodeBPETokenizer.from_files(vocab_file, merges_file)

        ids = tokenizer.encode_code("def binary_search", add_special_tokens=True)
        assert len(ids) > 0
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        assert "def" in decoded

        dsa_ids = tokenizer.encode_dsa_pair(
            problem_description="Binary Search", solution_code="def search(): pass"
        )
        assert len(dsa_ids) > 0


def test_pipeline_orchestrator() -> None:
    """Tests complete end-to-end PipelineOrchestrator execution."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        sample_file = raw_dir / "sample.py"
        sample_file.write_text("def add(a: int, b: int) -> int:\n    return a + b\n")

        cfg = EXP002Config(
            experiment_id="TEST_EXP_002",
            raw_data_dir=str(raw_dir),
            output_cache_dir=str(tmp_path / "cache"),
            tokenizer_dir=str(tmp_path / "tokenizer"),
            vocab_size=150,
            sequence_length=8,
            stride=8,
        )

        orchestrator = PipelineOrchestrator(config=cfg)
        summary = orchestrator.execute_pipeline()

        assert summary["status"] == "COMPLETED_SUCCESSFULLY"
        assert summary["files_processed"] >= 1
        assert summary["vocab_size"] > 0
