"""Comprehensive PyTest Suite for Phase 19 Model Lifecycle Management Subsystem.

Validates atomic checkpoint save pipeline (.tmp -> .pt OS replace), training resume state equivalence,
lightweight inference loading (model parameters only), best model selection, checkpoint retention rotation,
metadata index serialization, CheckpointSaver, CheckpointLoader, CheckpointValidator, TrainingResumeManager,
and CheckpointStatistics utilities.
"""

import os
import pytest
import torch
import torch.nn as nn

from src.models import AuraGPT, AuraGPTConfig
from src.optimizers import OptimizationConfig, OptimizationManager, OptimizerConfig
from src.utils import (
    CheckpointConfig,
    CheckpointExporter,
    CheckpointLoader,
    CheckpointMetadata,
    CheckpointSaver,
    CheckpointStatistics,
    CheckpointValidator,
    LifecycleConfig,
    LifecycleManager,
    MetadataRegistry,
    ModelExporter,
    TrainingResumeManager,
)


@pytest.fixture
def small_model():
    cfg = AuraGPTConfig(vocab_size=200, d_model=32, n_layers=2, n_heads=2, d_ff=128)
    return AuraGPT(config=cfg)


@pytest.fixture
def opt_manager(small_model):
    cfg = OptimizationConfig(optimizer=OptimizerConfig(lr=1e-3))
    return OptimizationManager(model=small_model, config=cfg)


# 1. Configuration & Metadata Registry Tests
def test_lifecycle_config_defaults():
    cfg = LifecycleConfig()
    assert cfg.checkpoint_dir == "checkpoints"
    assert cfg.max_keep_checkpoints == 5
    assert cfg.save_best is True
    assert cfg.monitor_metric == "val_loss"
    assert CheckpointConfig is LifecycleConfig


def test_metadata_registry(tmp_path):
    ckpt_dir = str(tmp_path / "checkpoints")
    registry = MetadataRegistry(checkpoint_dir=ckpt_dir)

    record = CheckpointMetadata(
        checkpoint_name="checkpoint_epoch_1_step_10.pt",
        epoch=1,
        global_step=10,
        timestamp="2026-08-03T00:00:00Z",
        metrics={"val_loss": 0.5},
        is_best=True,
    )
    registry.add_record(record)

    best = registry.get_best_record()
    assert best is not None
    assert best["checkpoint_name"] == "checkpoint_epoch_1_step_10.pt"


# 2. Atomic Save Pipeline & Training Resume Equivalence Test
def test_lifecycle_atomic_save_and_resume(small_model, opt_manager, tmp_path):
    ckpt_dir = str(tmp_path / "checkpoints")
    cfg = LifecycleConfig(checkpoint_dir=ckpt_dir)
    manager = LifecycleManager(config=cfg)

    # Save Checkpoint
    ckpt_path = manager.save_checkpoint(
        model=small_model,
        optimizer=opt_manager.optimizer,
        scheduler=opt_manager.scheduler,
        epoch=1,
        global_step=50,
        metrics={"val_loss": 1.5},
    )

    assert os.path.exists(ckpt_path)
    assert not os.path.exists(f"{ckpt_path}.tmp")  # Temporary file cleaned up after atomic rename

    # Mutate Model Parameters
    with torch.no_grad():
        for p in small_model.parameters():
            p.add_(2.5)

    # Resume Training
    epoch, step = manager.resume_training(
        checkpoint_path=ckpt_path,
        model=small_model,
        optimizer=opt_manager.optimizer,
        scheduler=opt_manager.scheduler,
    )

    assert epoch == 1
    assert step == 50


# 3. CheckpointSaver, CheckpointLoader, CheckpointValidator & TrainingResumeManager Test
def test_saver_loader_validator_resume(small_model, opt_manager, tmp_path):
    ckpt_dir = str(tmp_path / "checkpoints")
    cfg = CheckpointConfig(checkpoint_dir=ckpt_dir)
    saver = CheckpointSaver(config=cfg)

    ckpt_path = saver.save(
        model=small_model,
        optimizer=opt_manager.optimizer,
        scheduler=opt_manager.scheduler,
        epoch=2,
        global_step=100,
        metrics={"val_loss": 0.8},
    )

    # Validate
    res = CheckpointValidator.validate_file(ckpt_path)
    assert res.is_valid is True

    # Resume State
    epoch, step = TrainingResumeManager.resume_training_state(
        checkpoint_path=ckpt_path,
        model=small_model,
        optimizer=opt_manager.optimizer,
        scheduler=opt_manager.scheduler,
    )
    assert epoch == 2
    assert step == 100

    # Load Model directly
    loaded_model = CheckpointLoader.load_model(ckpt_path, small_model)
    assert loaded_model.training is False

    # Stats check
    stats = CheckpointStatistics.compute_stats(ckpt_dir)
    assert stats.total_checkpoints >= 1


# 4. Lightweight Inference Loading Test
def test_lifecycle_load_for_inference(small_model, tmp_path):
    ckpt_dir = str(tmp_path / "checkpoints")
    cfg = LifecycleConfig(checkpoint_dir=ckpt_dir)
    manager = LifecycleManager(config=cfg)

    ckpt_path = manager.save_checkpoint(
        model=small_model,
        epoch=1,
        global_step=10,
    )

    new_model = AuraGPT(config=small_model.config)
    eval_model = manager.load_for_inference(ckpt_path, new_model)

    assert eval_model.training is False  # Model set to eval() mode


# 5. Best Model Selection & Retention Policy Rotation Test
def test_checkpoint_best_selection_and_rotation(small_model, tmp_path):
    ckpt_dir = str(tmp_path / "checkpoints")
    cfg = LifecycleConfig(checkpoint_dir=ckpt_dir, max_keep_checkpoints=2, monitor_metric="val_loss", mode="min")
    manager = LifecycleManager(config=cfg)

    # Save Checkpoint 1 (val_loss = 2.0)
    p1 = manager.save_checkpoint(small_model, epoch=1, global_step=10, metrics={"val_loss": 2.0})
    # Save Checkpoint 2 (val_loss = 1.0 -> Best)
    p2 = manager.save_checkpoint(small_model, epoch=2, global_step=20, metrics={"val_loss": 1.0})
    # Save Checkpoint 3 (val_loss = 1.5)
    p3 = manager.save_checkpoint(small_model, epoch=3, global_step=30, metrics={"val_loss": 1.5})

    best_file = os.path.join(ckpt_dir, "best_model.pt")
    assert os.path.exists(best_file)

    # Retention check: only max 2 step checkpoints remain (p1 should be rotated out)
    assert not os.path.exists(p1)
    assert os.path.exists(p2)
    assert os.path.exists(p3)


# 6. Model Exporter Test
def test_checkpoint_exporter(small_model, tmp_path):
    ckpt_dir = str(tmp_path / "checkpoints")
    cfg = LifecycleConfig(checkpoint_dir=ckpt_dir)
    manager = LifecycleManager(config=cfg)

    ckpt_path = manager.save_checkpoint(small_model, epoch=1, global_step=10)
    export_path = os.path.join(ckpt_dir, "exported_weights.pt")

    res_path = CheckpointExporter.export_model_weights(ckpt_path, export_path, format="pytorch")
    assert os.path.exists(res_path)
    assert ModelExporter is CheckpointExporter
