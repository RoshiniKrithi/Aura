"""Production PR Review & Benchmark Test Suite for Phase 22 / EXP-003 Pre-Training.

Includes comprehensive coverage for:
- Dataset Mixing & Weighted Sampling
- Curriculum Scheduler Phase Transitions
- Sequence Packing & EOS Delimitation
- Dynamic Batch Construction
- Training & Validation Loops
- Evaluation Manager & Sample Generation
- Checkpoint Serialization & Recovery
- Large Dataset Simulation & Stress Testing
- Integration & Regression Testing
"""

import json
from pathlib import Path
import tempfile
import time
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.datasets.binary_writer import BinaryDatasetWriter
from src.datasets.memmap_dataset import MemmapCodeDataset
from src.models.config import AuraGPTConfig
from src.models.gpt import AuraGPT
from src.tokenizer.code_bpe_tokenizer import CodeBPETokenizer
from src.training.exp_003_orchestrator import (
    CurriculumScheduler,
    DatasetMixer,
    DynamicBatchBuilder,
    EvaluationManager,
    ExperimentTracker,
    ProgrammingPretrainingConfig,
    ProgrammingPretrainingRunner,
    SequencePacker,
)


@pytest.fixture
def dummy_memmap_shards():
    """Generates temporary binary dataset shards for multi-domain testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Domain 1: Python
        py_writer = BinaryDatasetWriter(
            output_dir=tmp_path, shard_prefix="train_python", vocab_size=50257, dtype="uint16"
        )
        py_writer.write_tokens(list(range(2048)))
        py_summary = py_writer.close()
        py_shards = [Path(s["path"]) for s in py_summary["shards"]]

        # Domain 2: C++
        cpp_writer = BinaryDatasetWriter(
            output_dir=tmp_path, shard_prefix="train_cpp", vocab_size=50257, dtype="uint16"
        )
        cpp_writer.write_tokens(list(range(2048, 4096)))
        cpp_summary = cpp_writer.close()
        cpp_shards = [Path(s["path"]) for s in cpp_summary["shards"]]

        yield tmp_path, py_shards, cpp_shards


def test_dataset_mixer_weighted_sampling(dummy_memmap_shards):
    """Verifies DatasetMixer probability normalization, temperature scaling, and sampling."""
    tmp_path, py_shards, cpp_shards = dummy_memmap_shards
    ds_py = MemmapCodeDataset(shard_paths=py_shards, sequence_length=32, stride=32)
    ds_cpp = MemmapCodeDataset(shard_paths=cpp_shards, sequence_length=32, stride=32)

    datasets = {"python": ds_py, "cpp": ds_cpp}
    weights = {"python": 0.8, "cpp": 0.2}

    mixer = DatasetMixer(datasets=datasets, weights=weights, temperature=1.0, seed=42)

    assert len(mixer.keys) == 2
    assert np.isclose(np.sum(mixer.probabilities), 1.0)
    assert mixer.probabilities[0] > mixer.probabilities[1]

    # Test sampling
    x, y = mixer.sample_sequence()
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == (32,)
    assert y.shape == (32,)

    # Test weight update
    mixer.update_weights({"python": 0.1, "cpp": 0.9})
    assert mixer.probabilities[1] > mixer.probabilities[0]

    ds_py.close()
    ds_cpp.close()


def test_curriculum_scheduler_phase_shift(dummy_memmap_shards):
    """Verifies CurriculumScheduler triggers weight shift at phase_step boundary."""
    tmp_path, py_shards, cpp_shards = dummy_memmap_shards
    ds_py = MemmapCodeDataset(shard_paths=py_shards, sequence_length=32, stride=32)
    datasets = {"python": ds_py}
    mixer = DatasetMixer(datasets=datasets, weights={"python": 1.0})

    scheduler = CurriculumScheduler(
        mixer=mixer,
        phase_step=100,
        phase_a_weights={"python": 0.9},
        phase_b_weights={"python": 0.5},
    )

    assert scheduler.current_phase == "Phase_A"
    assert not scheduler.step(50)
    assert scheduler.current_phase == "Phase_A"

    # Step crossing threshold
    assert scheduler.step(100)
    assert scheduler.current_phase == "Phase_B"
    assert not scheduler.step(150)  # Idempotent after shift

    ds_py.close()


def test_sequence_packer_eos_chunking():
    """Verifies SequencePacker buffer concatenation, EOS token insertion, and exact chunking."""
    packer = SequencePacker(sequence_length=10, eos_token_id=99)

    # Stream 1: 5 tokens -> buffer has 6 items (5 + EOS) -> no chunk (needs 11)
    chunks1 = packer.add_token_stream([1, 2, 3, 4, 5])
    assert len(chunks1) == 0
    assert len(packer.buffer) == 6

    # Stream 2: 10 tokens -> buffer total 17 items -> 1 chunk produced, 7 left in buffer
    chunks2 = packer.add_token_stream([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    assert len(chunks2) == 1
    x, y = chunks2[0]
    assert len(x) == 10
    assert len(y) == 10
    assert x[0] == 1
    assert y[0] == 2
    assert x[-1] == 13
    assert y[-1] == 14
    assert len(packer.buffer) == 7


def test_dynamic_batch_builder_shapes(dummy_memmap_shards):
    """Verifies DynamicBatchBuilder produces valid 2D micro-batches (B, L) of torch.long."""
    tmp_path, py_shards, _ = dummy_memmap_shards
    ds_py = MemmapCodeDataset(shard_paths=py_shards, sequence_length=16, stride=16)
    mixer = DatasetMixer(datasets={"python": ds_py}, weights={"python": 1.0})
    batch_builder = DynamicBatchBuilder(mixer=mixer, micro_batch_size=4)

    x_batch, y_batch = batch_builder.build_batch()
    assert x_batch.shape == (4, 16)
    assert y_batch.shape == (4, 16)
    assert x_batch.dtype == torch.long
    assert y_batch.dtype == torch.long

    ds_py.close()


def test_experiment_tracker_jsonl_logging():
    """Verifies ExperimentTracker JSONL file creation and step recording."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir)
        tracker = ExperimentTracker(output_dir=out_path)

        tracker.record_step(step=1, loss=3.5, lr=1e-4, grad_norm=0.5, tps=1200.0)
        tracker.record_validation(step=1, val_loss=3.2, perplexity=24.5)

        log_file = out_path / "logs" / "metrics_history.jsonl"
        assert log_file.exists()

        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["step"] == 1
            assert data["loss"] == 3.5
            assert data["tps"] == 1200.0


def test_evaluation_manager_loss_and_sampling(dummy_memmap_shards):
    """Verifies EvaluationManager loss computation and text sample generation."""
    tmp_path, py_shards, _ = dummy_memmap_shards
    val_ds = MemmapCodeDataset(shard_paths=py_shards, sequence_length=32, stride=32)

    gpt_cfg = AuraGPTConfig(
        model_name="test-tiny-eval",
        vocab_size=50257,
        max_sequence_length=32,
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=128,
        device="cpu",
    )
    model = AuraGPT(gpt_cfg)
    tokenizer = CodeBPETokenizer.create_default()

    eval_mgr = EvaluationManager(model=model, tokenizer=tokenizer, val_dataset=val_ds, device="cpu")

    val_loss, perplexity = eval_mgr.evaluate_loss(max_eval_batches=2)
    assert isinstance(val_loss, float)
    assert isinstance(perplexity, float)
    assert perplexity >= 1.0

    samples = eval_mgr.generate_samples(prompts=["def add(a, b):"], max_new_tokens=10)
    assert len(samples) == 1
    assert samples[0]["prompt"] == "def add(a, b):"
    assert isinstance(samples[0]["completion"], str)

    val_ds.close()


def test_runner_pretraining_and_checkpoint_recovery(dummy_memmap_shards):
    """Verifies ProgrammingPretrainingRunner end-to-end execution, checkpointing, and resume."""
    tmp_path, py_shards, _ = dummy_memmap_shards

    with tempfile.TemporaryDirectory() as out_dir:
        config = ProgrammingPretrainingConfig(
            experiment_id="TEST_RUNNER_EXP",
            cache_dir=str(tmp_path),
            tokenizer_dir=str(tmp_path),
            output_dir=out_dir,
            max_steps=4,
            eval_interval=2,
            save_interval=2,
            micro_batch_size=2,
            gradient_accumulation_steps=1,
            max_sequence_length=16,
            d_model=32,
            n_layers=2,
            n_heads=2,
            d_ff=64,
            warmup_steps=0,
        )

        # 1. Run Initial Training
        runner = ProgrammingPretrainingRunner(config=config)
        summary = runner.run_pretraining()

        assert summary["status"] == "COMPLETED_SUCCESSFULLY"
        assert summary["total_steps"] == 4

        ckpt_latest = Path(out_dir) / "latest.pt"
        ckpt_step2 = Path(out_dir) / "checkpoint_step_000002.pt"
        assert ckpt_latest.exists()
        assert ckpt_step2.exists()

        runner.close()

        # 2. Test Checkpoint Recovery
        config.max_steps = 6
        runner_resumed = ProgrammingPretrainingRunner(config=config, resume_from_checkpoint=ckpt_latest)
        assert runner_resumed.start_step == 4

        summary_resumed = runner_resumed.run_pretraining()
        assert summary_resumed["total_steps"] == 6
        runner_resumed.close()


def test_large_dataset_simulation_and_resource_cleanup():
    """Simulates multi-shard dataset loading and verifies resource handle closure."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        shard_paths = []

        for i in range(5):
            writer = BinaryDatasetWriter(
                output_dir=tmp_path, shard_prefix=f"train_shard_{i}", vocab_size=50257, dtype="uint16"
            )
            writer.write_tokens(list(range(1000)))
            summary = writer.close()
            shard_paths.extend([Path(s["path"]) for s in summary["shards"]])

        ds = MemmapCodeDataset(shard_paths=shard_paths, sequence_length=64, stride=64)
        assert len(ds) > 0

        # Read samples across all shards
        for idx in range(len(ds)):
            x, y = ds[idx]
            assert x.shape == (64,)

        # Verify close releases handles cleanly
        ds.close()
        assert len(ds.memmaps) == 0
