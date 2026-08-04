"""Comprehensive PyTest Suite for Phase 16 Optimization Subsystem.

Validates AdamW weight decay parameter filtering (2D vs 1D parameters),
CosineAnnealingWithWarmupLR learning rate curves, GradientManager norm clipping (C=1.0),
gradient accumulation triggers, OptimizationManager state_dict serialization, and factory builders.
"""

import math
import pytest
import torch
import torch.nn as nn

from src.optimizers import (
    CheckpointUtilities,
    GradientManager,
    OptimizationConfig,
    OptimizationManager,
    OptimizationStatistics,
    OptimizerConfig,
    OptimizerFactory,
    WeightDecayUtilities,
)
from src.schedulers import (
    CosineAnnealingWithWarmupLR,
    SchedulerConfig,
    SchedulerFactory,
)


class DummyModel(nn.Module):
    """Simple 2-layer linear network for testing optimization updates."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20, bias=True)
        self.ln = nn.LayerNorm(20)
        self.fc2 = nn.Linear(20, 5, bias=True)

    def forward(self, x):
        return self.fc2(self.ln(self.fc1(x)))


@pytest.fixture
def dummy_model():
    return DummyModel()


# 1. Configuration Schema Tests
def test_optimization_config_defaults():
    cfg = OptimizationConfig()
    assert cfg.optimizer.name == "adamw"
    assert cfg.optimizer.weight_decay == 0.1
    assert cfg.scheduler.name == "cosine_warmup"
    assert cfg.max_grad_norm == 1.0
    assert cfg.gradient_accumulation_steps == 1


# 2. Parameter Group Weight Decay Filtering Test
def test_parameter_group_filtering(dummy_model):
    param_groups = OptimizerFactory.prepare_parameter_groups(
        dummy_model, weight_decay=0.1, filter_weight_decay=True
    )

    assert len(param_groups) == 2
    # Group 0: 2D linear weights (fc1.weight, fc2.weight) -> weight_decay = 0.1
    assert param_groups[0]["weight_decay"] == 0.1
    # Group 1: 1D biases and LayerNorm weights (fc1.bias, ln.weight, ln.bias, fc2.bias) -> weight_decay = 0.0
    assert param_groups[1]["weight_decay"] == 0.0


# 3. Cosine Annealing with Warmup Learning Rate Curve Test
def test_cosine_warmup_lr_curve(dummy_model):
    optimizer = torch.optim.AdamW(dummy_model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingWithWarmupLR(
        optimizer=optimizer,
        warmup_steps=100,
        max_steps=1000,
        min_lr=1e-5,
    )

    # Step 0: Warmup start (LR should be 0.0)
    lrs_step0 = scheduler.get_lr()
    assert lrs_step0[0] == 0.0

    # Step 50: Mid warmup (LR should be 50% of peak 1e-3 = 5e-4)
    scheduler.last_epoch = 50
    lrs_step50 = scheduler.get_lr()
    assert pytest.approx(lrs_step50[0], rel=1e-3) == 5e-4

    # Step 100: End of warmup (LR should be peak 1e-3)
    scheduler.last_epoch = 100
    lrs_step100 = scheduler.get_lr()
    assert pytest.approx(lrs_step100[0], rel=1e-3) == 1e-3

    # Step 1000: End of decay (LR should be floor min_lr 1e-5)
    scheduler.last_epoch = 1000
    lrs_step1000 = scheduler.get_lr()
    assert pytest.approx(lrs_step1000[0], rel=1e-3) == 1e-5

    # Step 1500: Beyond max steps (LR should remain min_lr 1e-5)
    scheduler.last_epoch = 1500
    lrs_step1500 = scheduler.get_lr()
    assert lrs_step1500[0] == 1e-5


# 4. Gradient Manager Norm Clipping Test
def test_gradient_manager_clipping(dummy_model):
    grad_mgr = GradientManager(max_grad_norm=1.0, accumulation_steps=1)

    x = torch.randn(4, 10)
    out = dummy_model(x).sum() * 1000.0  # Force large gradients
    out.backward()

    # Pre-clipping gradient norm should be large
    grad_norm = grad_mgr.clip_grad_norm(dummy_model.parameters())
    assert grad_norm > 1.0

    # Post-clipping gradient norm of parameters should be <= 1.0
    post_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in dummy_model.parameters() if p.grad is not None], 1.0
    )
    assert float(post_norm) <= 1.001


# 5. Gradient Accumulation Trigger Test
def test_gradient_accumulation_triggers():
    grad_mgr = GradientManager(accumulation_steps=4)

    assert grad_mgr.should_step(0) is False  # Step 1/4
    assert grad_mgr.should_step(1) is False  # Step 2/4
    assert grad_mgr.should_step(2) is False  # Step 3/4
    assert grad_mgr.should_step(3) is True   # Step 4/4 (Trigger update)


# 6. OptimizationManager End-to-End Execution Test
def test_optimization_manager_end_to_end(dummy_model):
    cfg = OptimizationConfig(
        optimizer=OptimizerConfig(name="adamw", lr=1e-3),
        scheduler=SchedulerConfig(name="cosine_warmup", warmup_steps=10, max_steps=100),
        max_grad_norm=1.0,
        gradient_accumulation_steps=2,
    )

    opt_mgr = OptimizationManager(model=dummy_model, config=cfg)

    # Micro-step 0 (Accumulate - no optimizer step)
    x0 = torch.randn(2, 10)
    loss0 = dummy_model(x0).sum()
    loss0.backward()
    did_step0, lr0, norm0 = opt_mgr.step(micro_step=0)

    assert did_step0 is False

    # Micro-step 1 (Accumulation target 2 met -> Trigger optimizer step)
    x1 = torch.randn(2, 10)
    loss1 = dummy_model(x1).sum()
    loss1.backward()
    did_step1, lr1, norm1 = opt_mgr.step(micro_step=1)

    assert did_step1 is True
    assert norm1 >= 0.0
    assert opt_mgr.global_step == 1


# 7. Checkpointing State Dict Serialization Test
def test_optimization_manager_checkpointing(dummy_model):
    opt_mgr = OptimizationManager(model=dummy_model)
    opt_mgr.global_step = 42
    opt_mgr.micro_step = 84

    state_dict = opt_mgr.state_dict()
    assert state_dict["global_step"] == 42
    assert state_dict["micro_step"] == 84

    new_opt_mgr = OptimizationManager(model=dummy_model)
    new_opt_mgr.load_state_dict(state_dict)

    assert new_opt_mgr.global_step == 42
    assert new_opt_mgr.micro_step == 84


# 8. Factory Builders Test
def test_factories_creation(dummy_model):
    opt = OptimizerFactory.create_optimizer(dummy_model, name="adamw")
    assert isinstance(opt, torch.optim.AdamW)

    sched = SchedulerFactory.create_scheduler(opt, name="cosine_warmup")
    assert isinstance(sched, CosineAnnealingWithWarmupLR)


# 9. WeightDecayUtilities & CheckpointUtilities Test
def test_utilities_and_statistics(dummy_model):
    decay, no_decay = WeightDecayUtilities.filter_weight_decay_params(dummy_model, weight_decay=0.1)
    assert len(decay) == 2  # fc1.weight, fc2.weight
    assert len(no_decay) == 4  # fc1.bias, ln.weight, ln.bias, fc2.bias

    opt = torch.optim.AdamW(dummy_model.parameters(), lr=1e-3)
    sched = CosineAnnealingWithWarmupLR(opt)

    ckpt = CheckpointUtilities.create_optimization_checkpoint(opt, sched, global_step=10, micro_step=20)
    g_step, m_step = CheckpointUtilities.restore_optimization_checkpoint(ckpt, opt, sched)
    assert g_step == 10
    assert m_step == 20

    stats = OptimizationStatistics.compute_stats(dummy_model, current_lr=1e-3, global_step=10)
    assert stats.decay_params_count > 0
    assert stats.no_decay_params_count > 0

