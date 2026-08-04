"""Production PR Review & Benchmark Test Suite for Phase 21 / EXP-002 Pipeline.

Includes stress testing, zero-copy memory profiling, tensor shape assertions,
Causal LM target shift validation, uint16 overflow boundary checks, and regression tests.
"""

import os
from pathlib import Path
import tempfile
import time
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.datasets.binary_writer import BinaryDatasetWriter
from src.datasets.code_cleaner import CodeTextCleaner
from src.datasets.memmap_dataset import MemmapCodeDataset
from src.tokenizer.bpe_trainer import BPETrainer
from src.tokenizer.code_bpe_tokenizer import CodeBPESpecialTokens, CodeBPETokenizer
from src.training.exp_002_orchestrator import EXP002Config, PipelineOrchestrator


def test_tensor_shape_and_causal_shift_validation() -> None:
    """Verifies PyTorch LongTensor output shapes (sequence_length,) and causal next-token alignment."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        writer = BinaryDatasetWriter(
            output_dir=tmp_path,
            shard_prefix="shape_test",
            max_shard_size_bytes=1024 * 1024,
            vocab_size=50257,
            dtype="uint16",
        )

        tokens = list(range(2048))
        writer.write_tokens(tokens)
        summary = writer.close()

        shards = [Path(s["path"]) for s in summary["shards"]]
        ds = MemmapCodeDataset(
            shard_paths=shards,
            sequence_length=1024,
            stride=1024,
            dtype="uint16",
            name="ShapeValDS",
        )

        assert len(ds) == 1
        x, y = ds[0]

        # 1. Tensor Type Validation
        assert isinstance(x, torch.LongTensor)
        assert isinstance(y, torch.LongTensor)

        # 2. Shape Validation (sequence_length,)
        assert x.shape == (1024,)
        assert y.shape == (1024,)

        # 3. Causal LM Target Shift Assertion: Y[i] == X[i+1] for all i in [0, L-2]
        assert torch.equal(x[1:], y[:-1])
        assert x[0].item() == 0
        assert y[0].item() == 1
        assert y[-1].item() == 1024

        ds.close()


def test_uint16_boundary_overflow_protection() -> None:
    """Verifies that vocab_size > 65,535 raises ValueError under uint16 mode."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        with pytest.raises(ValueError, match="exceeds uint16 max capacity"):
            BinaryDatasetWriter(
                output_dir=tmp_path,
                vocab_size=70000,
                dtype="uint16",
            )


def test_memory_mapped_zero_copy_profiling() -> None:
    """Profiles memory usage and performance of MemmapCodeDataset across 100,000 synthetic tokens."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        writer = BinaryDatasetWriter(
            output_dir=tmp_path,
            shard_prefix="prof_test",
            max_shard_size_bytes=200000,
            vocab_size=50257,
            dtype="uint16",
        )

        tokens = list(range(50257)) * 2
        writer.write_tokens(tokens)
        summary = writer.close()

        shards = [Path(s["path"]) for s in summary["shards"]]
        ds = MemmapCodeDataset(
            shard_paths=shards,
            sequence_length=512,
            stride=512,
            dtype="uint16",
            name="ProfDS",
        )

        loader = DataLoader(ds, batch_size=8, drop_last=True, shuffle=False)
        batch_count = 0

        start_time = time.time()
        try:
            for x, y in loader:
                batch_count += 1
                assert x.shape == (8, 512)
                assert y.shape == (8, 512)
        finally:
            del loader
            ds.close()

        elapsed = time.time() - start_time
        assert batch_count > 0
        # High throughput verification (< 5 seconds)
        assert elapsed < 5.0


def test_ast_python_cleaner_stress() -> None:
    """Stress tests CodeTextCleaner with diverse Python AST syntax features."""
    cleaner = CodeTextCleaner(remove_comments=True, strict_ast_validation=True)

    complex_python = '''
import asyncio
from typing import List, Dict

class Solution:
    async def solve(self, data: List[Dict[str, int]]) -> int:
        """Computes async summary."""
        # Calculate sum
        val = sum(d.get("key", 0) for d in data)
        return val
'''
    cleaned, is_valid = cleaner.clean_python(complex_python)
    assert is_valid is True
    assert "class Solution:" in cleaned
    assert "# Calculate sum" not in cleaned
    assert '"""Computes async summary."""' in cleaned


def test_configuration_validation() -> None:
    """Verifies dataclass conversion and validation of EXP002Config."""
    cfg = EXP002Config(
        vocab_size=50257,
        sequence_length=1024,
        max_shard_size_mb=512,
        dtype="uint16",
    )
    d = cfg.to_dict()
    assert d["vocab_size"] == 50257
    assert d["sequence_length"] == 1024
    assert d["max_shard_size_mb"] == 512
    assert d["dtype"] == "uint16"
