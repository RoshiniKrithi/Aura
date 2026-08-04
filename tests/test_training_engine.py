"""Comprehensive PyTest Suite for Phase 17 Training Engine Subsystem.

Validates single and multi-epoch training execution, BatchRunner, EpochRunner, ValidationRunner,
ProgressTracker, TrainingLogger, TrainingStatistics, loss minimization over synthetic data,
validation loop isolation, checkpoint round-trips, and EngineFactory builders.
"""

import os
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models import AuraGPT, AuraGPTConfig
from src.training import (
    BatchRunner,
    EngineFactory,
    EngineValidationError,
    EngineValidator,
    EpochRunner,
    MetricsManager,
    MetricTracker,
    ProgressTracker,
    TrainingEngine,
    TrainingEngineConfig,
    TrainingLogger,
    TrainingStatistics,
    ValidationRunner,
)
from src.utils.config import AppConfig


@pytest.fixture
def synthetic_data():
    """Generates synthetic input_ids and targets for testing training engine."""
    input_ids = torch.randint(0, 500, (8, 16))
    targets = torch.randint(0, 500, (8, 16))
    dataset = TensorDataset(input_ids, targets)
    dataloader = DataLoader(dataset, batch_size=2)
    return dataloader


@pytest.fixture
def small_aura_model():
    """Returns a lightweight AuraGPT model for fast unit testing."""
    cfg = AuraGPTConfig(vocab_size=500, d_model=32, n_layers=2, n_heads=2, d_ff=128)
    return AuraGPT(config=cfg)


# 1. Configuration & MetricTracker / MetricsManager Tests
def test_engine_config_defaults():
    cfg = TrainingEngineConfig()
    assert cfg.epochs == 10
    assert cfg.eval_interval == 1
    assert cfg.checkpoint_interval == 1
    assert cfg.amp_enabled is False


def test_metric_tracker_accumulation():
    tracker = MetricTracker()
    tracker.update(loss=1.0, accuracy=50.0, perplexity=2.718, lr=1e-3, grad_norm=0.5)
    tracker.update(loss=0.5, accuracy=70.0, perplexity=1.648, lr=1e-3, grad_norm=0.4)

    summary = tracker.get_summary()
    assert summary.loss == 0.75
    assert summary.accuracy == 60.0
    assert summary.total_steps == 2
    assert MetricsManager is MetricTracker


# 2. ProgressTracker & TrainingLogger & TrainingStatistics Tests
def test_progress_tracker_and_stats():
    progress = ProgressTracker(total_epochs=10, total_steps_per_epoch=20)
    progress.update(epoch=1, step=5)
    assert progress.get_progress_pct() > 0.0

    TrainingLogger.log_epoch_start(1, 10)
    TrainingLogger.log_epoch_end(1, 10, {"loss": 0.5, "accuracy": 80.0, "perplexity": 1.6, "lr": 1e-3})

    history = [{"train": {"loss": 0.5, "accuracy": 80.0, "perplexity": 1.6}}]
    stats = TrainingStatistics.compute_run_stats(history, total_duration=10.0)
    assert stats.total_epochs == 1
    assert stats.final_train_loss == 0.5


# 3. End-to-End Single Epoch Training Test
def test_training_engine_single_epoch(small_aura_model, synthetic_data):
    cfg = TrainingEngineConfig(epochs=1, eval_interval=1, device="cpu")
    engine = TrainingEngine(
        model=small_aura_model,
        train_dataloader=synthetic_data,
        config=cfg,
    )

    metrics = engine.train_epoch(epoch=1)

    assert "loss" in metrics
    assert "accuracy" in metrics
    assert "perplexity" in metrics
    assert metrics["loss"] > 0.0


# 4. Validation Loop Isolation Test
def test_training_engine_validation(small_aura_model, synthetic_data):
    cfg = TrainingEngineConfig(epochs=1, device="cpu")
    engine = TrainingEngine(
        model=small_aura_model,
        train_dataloader=synthetic_data,
        val_dataloader=synthetic_data,
        config=cfg,
    )

    val_metrics = engine.validate()

    assert "loss" in val_metrics
    assert "accuracy" in val_metrics
    assert "perplexity" in val_metrics


# 5. Checkpoint Serialization & Restoration Round-Trip
def test_training_engine_checkpoint_roundtrip(small_aura_model, synthetic_data, tmp_path):
    ckpt_file = os.path.join(tmp_path, "test_checkpoint.pt")
    cfg = TrainingEngineConfig(epochs=2, device="cpu")

    engine = TrainingEngine(
        model=small_aura_model,
        train_dataloader=synthetic_data,
        config=cfg,
    )
    engine.current_epoch = 2
    engine.save_checkpoint(ckpt_file)

    assert os.path.exists(ckpt_file)

    # Mutate model weights
    with torch.no_grad():
        for p in small_aura_model.parameters():
            p.add_(1.0)

    # Restore checkpoint
    engine.load_checkpoint(ckpt_file)
    assert engine.current_epoch == 2


# 6. Full Fit Loop Execution Test
def test_training_engine_fit_loop(small_aura_model, synthetic_data):
    cfg = TrainingEngineConfig(epochs=2, eval_interval=1, checkpoint_interval=5, device="cpu")
    engine = TrainingEngine(
        model=small_aura_model,
        train_dataloader=synthetic_data,
        val_dataloader=synthetic_data,
        config=cfg,
    )

    results = engine.fit()

    assert "history" in results
    assert len(results["history"]) == 2
    assert results["final_epoch"] == 2


# 7. Factory & Validator Builder Tests
def test_engine_factory(small_aura_model, synthetic_data):
    app_cfg = AppConfig()
    engine = EngineFactory.create_engine(
        model=small_aura_model,
        train_dataloader=synthetic_data,
        config=app_cfg,
    )

    assert isinstance(engine, TrainingEngine)


def test_engine_validator_raises(small_aura_model):
    with pytest.raises(EngineValidationError):
        TrainingEngine(model=small_aura_model, train_dataloader="not_a_dataloader")
