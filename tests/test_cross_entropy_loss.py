"""Comprehensive PyTest Suite for Phase 15 Training Objective & Cross-Entropy Loss Subsystem.

Validates sequence label shifting (logits[:, :-1, :] vs targets[:, 1:]), ignore_index padding masks,
differentiable loss scalar output, accuracy, perplexity calculation, label smoothing,
validator NaN/Inf checks, and loss factory builders.
"""

import math
import pytest
import torch

from src.losses import (
    CrossEntropyLoss,
    CrossEntropyLossConfig,
    LossFactory,
    LossStatistics,
    LossValidationError,
    LossValidator,
)


@pytest.fixture
def loss_module():
    """Returns an initialized default CrossEntropyLoss module."""
    return CrossEntropyLoss(config=CrossEntropyLossConfig(ignore_index=-1))


# 1. Configuration Schema Tests
def test_loss_config_defaults():
    cfg = CrossEntropyLossConfig()
    assert cfg.ignore_index == -1
    assert cfg.label_smoothing == 0.0
    assert cfg.reduction == "mean"
    assert cfg.compute_accuracy is True
    assert cfg.compute_perplexity is True


# 2. Sequence Label Shifting Test
def test_sequence_shifting_alignment(loss_module):
    # Batch=1, Sequence=3 (tokens: 0, 1, 2), Vocab=10
    # Position 0 predicts Token 1, Position 1 predicts Token 2
    logits = torch.zeros(1, 3, 10, requires_grad=True)
    targets = torch.tensor([[100, 5, 8]])  # Token 1 is 5, Token 2 is 8

    # Set logits at pos 0 to strongly predict token 5
    with torch.no_grad():
        logits[0, 0, 5] = 10.0
        # Set logits at pos 1 to strongly predict token 8
        logits[0, 1, 8] = 10.0

    loss, metrics = loss_module(logits, targets)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.item() < 0.1  # Low loss because predictions align with shifted targets
    assert metrics["accuracy"] == 100.0
    assert metrics["perplexity"] == pytest.approx(math.exp(loss.item()), rel=1e-3)


# 3. Ignore Index Padding Mask Verification
def test_ignore_index_padding_mask(loss_module):
    logits = torch.randn(2, 4, 10, requires_grad=True)
    targets = torch.tensor([
        [1, 2, 3, -1],   # Target 3 is -1 (ignore)
        [4, 5, -1, -1],  # Targets 2 & 3 are -1 (ignore)
    ])

    loss, metrics = loss_module(logits, targets)

    # Mutate logits at ignored position [0, 3] and check that loss remains unchanged
    logits_mod = logits.clone().detach()
    logits_mod[0, 3, :] = 999.0  # Extreme values at ignored position
    logits_mod.requires_grad = True

    loss_mod, _ = loss_module(logits_mod, targets)

    assert torch.isclose(loss, loss_mod, atol=1e-5)


# 4. Label Smoothing Test
def test_label_smoothing():
    loss_smooth = CrossEntropyLoss(
        config=CrossEntropyLossConfig(ignore_index=-1, label_smoothing=0.1)
    )

    logits = torch.randn(2, 4, 10)
    targets = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

    loss, metrics = loss_smooth(logits, targets)
    assert loss.item() > 0.0


# 5. Gradient Propagation Test
def test_loss_gradient_propagation(loss_module):
    logits = torch.randn(2, 4, 10, requires_grad=True)
    targets = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

    loss, _ = loss_module(logits, targets)
    loss.backward()

    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()


# 6. Validator NaN & Inf Checks
def test_loss_validator_nan_inf():
    validator = LossValidator()

    logits_nan = torch.randn(2, 4, 10)
    logits_nan[0, 0, 0] = float("nan")
    targets = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

    res_nan = validator.validate_inputs(logits_nan, targets)
    assert res_nan.is_valid is False
    assert res_nan.has_nan is True


def test_loss_raises_on_invalid_input(loss_module):
    bad_logits = torch.randn(2, 4, 10)
    bad_targets = torch.tensor([[1, 2, 3]])  # Shape mismatch (sequence len 3 vs 4)

    with pytest.raises(LossValidationError):
        loss_module(bad_logits, bad_targets)


# 7. Loss Factory Creation Test
def test_loss_factory_creation():
    loss = LossFactory.create_loss(ignore_index=-1, label_smoothing=0.05)
    assert isinstance(loss, CrossEntropyLoss)
    assert loss.ignore_index == -1
    assert loss.label_smoothing == 0.05


# 8. Statistics Extractor Test
def test_loss_statistics():
    logits = torch.randn(2, 4, 10)
    targets = torch.tensor([[1, 2, 3, -1], [4, 5, -1, -1]])

    # Pass shifted targets targets[:, 1:]
    stats = LossStatistics.compute_stats(logits[:, :-1, :], targets[:, 1:], ignore_index=-1)

    # Shifted targets shape is (2, 3) = 6 total tokens
    # Row 0 shifted: [2, 3, -1] -> 2 valid, 1 ignored
    # Row 1 shifted: [5, -1, -1] -> 1 valid, 2 ignored
    assert stats.valid_tokens == 3
    assert stats.ignored_tokens == 3


# 9. Parametrized Batch & Sequence Length Shapes
@pytest.mark.parametrize("b_size", [1, 4])
@pytest.mark.parametrize("seq_len", [2, 16, 64])
def test_parametrized_loss_shapes(b_size, seq_len):
    loss_mod = CrossEntropyLoss()

    logits = torch.randn(b_size, seq_len, 100)
    targets = torch.randint(0, 100, (b_size, seq_len))

    loss, metrics = loss_mod(logits, targets)

    assert isinstance(loss, torch.Tensor)
    assert "loss" in metrics
    assert "accuracy" in metrics
    assert "perplexity" in metrics
