"""Comprehensive PyTest Suite for Experiment EXP-003.

Validates DatasetMixer, CurriculumScheduler, SequencePacker, DynamicBatchBuilder,
ExperimentTracker, EvaluationManager, and ProgrammingPretrainingRunner.
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch

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
def temp_cache_dir():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        tmp_path = Path(tmpdir)
        writer = BinaryDatasetWriter(
            output_dir=tmp_path,
            shard_prefix="train",
            vocab_size=50257,
            dtype="uint16",
        )
        writer.write_tokens(list(range(2048)))
        summary = writer.close()
        yield tmp_path


def test_sequence_packer():
    packer = SequencePacker(sequence_length=16, eos_token_id=3)
    tokens = list(range(10, 50))
    packed = packer.add_token_stream(tokens)

    assert len(packed) >= 2
    x, y = packed[0]
    assert len(x) == 16
    assert len(y) == 16
    assert (y[:15] == x[1:16]).all()


def test_dataset_mixer(temp_cache_dir):
    shards = list(temp_cache_dir.glob("train_*.bin"))
    ds1 = MemmapCodeDataset(shard_paths=shards, sequence_length=32, stride=32)
    ds2 = MemmapCodeDataset(shard_paths=shards, sequence_length=32, stride=32)

    mixer = DatasetMixer(
        datasets={"py": ds1, "cpp": ds2},
        weights={"py": 0.8, "cpp": 0.2},
    )

    x, y = mixer.sample_sequence()
    assert x.shape == (32,)
    assert y.shape == (32,)
    ds1.close()
    ds2.close()


def test_curriculum_scheduler(temp_cache_dir):
    shards = list(temp_cache_dir.glob("train_*.bin"))
    ds = MemmapCodeDataset(shard_paths=shards, sequence_length=32, stride=32)
    mixer = DatasetMixer(datasets={"py": ds}, weights={"py": 1.0})

    scheduler = CurriculumScheduler(mixer=mixer, phase_step=50)
    assert scheduler.current_phase == "Phase_A"

    shifted = scheduler.step(global_step=50)
    assert shifted is True
    assert scheduler.current_phase == "Phase_B"
    ds.close()


def test_dynamic_batch_builder(temp_cache_dir):
    shards = list(temp_cache_dir.glob("train_*.bin"))
    ds = MemmapCodeDataset(shard_paths=shards, sequence_length=32, stride=32)
    mixer = DatasetMixer(datasets={"py": ds}, weights={"py": 1.0})

    builder = DynamicBatchBuilder(mixer=mixer, micro_batch_size=4)
    x_batch, y_batch = builder.build_batch()

    assert x_batch.shape == (4, 32)
    assert y_batch.shape == (4, 32)
    ds.close()


def test_evaluation_manager(temp_cache_dir):
    shards = list(temp_cache_dir.glob("train_*.bin"))
    ds = MemmapCodeDataset(shard_paths=shards, sequence_length=32, stride=32)

    gpt_cfg = AuraGPTConfig(
        model_name="test-tiny",
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

    eval_mgr = EvaluationManager(model=model, tokenizer=tokenizer, val_dataset=ds, device="cpu")
    val_loss, val_ppl = eval_mgr.evaluate_loss(max_eval_batches=2)

    assert isinstance(val_loss, float)
    assert isinstance(val_ppl, float)
    assert val_ppl >= 1.0
    ds.close()


def test_programming_pretraining_runner(temp_cache_dir):
    with tempfile.TemporaryDirectory() as outdir:
        config = ProgrammingPretrainingConfig(
            experiment_id="TEST_EXP_003",
            cache_dir=str(temp_cache_dir),
            tokenizer_dir=str(temp_cache_dir),
            output_dir=outdir,
            max_steps=5,
            eval_interval=5,
            save_interval=5,
            micro_batch_size=2,
            gradient_accumulation_steps=1,
            max_sequence_length=32,
            d_model=64,
            n_layers=2,
            n_heads=2,
            d_ff=128,
            warmup_steps=0,
        )

        runner = ProgrammingPretrainingRunner(config=config)
        summary = runner.run_pretraining()

        assert summary["status"] == "COMPLETED_SUCCESSFULLY"
        assert summary["total_steps"] == 5
        assert Path(outdir, "latest.pt").exists()
        runner.close()
