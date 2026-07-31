"""Comprehensive PyTest Suite for Aura Transformer Feed Forward Network (FFN / MLP) Module.

Validates FeedForwardNetwork forward pass shapes, activation function variants (GELU, ReLU, SiLU, SwiGLU),
weight initializers, gradient propagation, input validator NaN/Inf detection, statistics, and factory creation.
"""

import pytest
import torch
import torch.nn as nn

from src.ffn import (
    FeedForwardConfig,
    FeedForwardFactory,
    FeedForwardInitializer,
    FeedForwardNetwork,
    FeedForwardStatistics,
    FeedForwardValidationError,
    FeedForwardValidator,
)
from src.utils.config import AppConfig


@pytest.fixture
def ffn_config():
    """Returns a test FeedForwardConfig."""
    return FeedForwardConfig(d_model=64, expansion_factor=4, dropout=0.0, activation="gelu")


@pytest.fixture
def sample_ffn(ffn_config):
    """Returns an initialized FeedForwardNetwork module."""
    return FeedForwardNetwork(config=ffn_config)


# 1. Configuration Tests
def test_feed_forward_config(ffn_config):
    assert ffn_config.d_model == 64
    assert ffn_config.hidden_dim == 256
    assert ffn_config.activation == "gelu"


# 2. Forward Pass Tests (3D & 2D)
def test_ffn_forward_3d(sample_ffn):
    x = torch.randn(2, 8, 64)
    out = sample_ffn(x)
    assert out.shape == (2, 8, 64)
    assert out.dtype == torch.float32


def test_ffn_forward_2d(sample_ffn):
    x = torch.randn(8, 64)
    out = sample_ffn(x)
    assert out.shape == (8, 64)


# 3. Activation Functions Test (GELU, ReLU, SiLU, SwiGLU)
@pytest.mark.parametrize("act", ["gelu", "relu", "silu", "swiglu"])
def test_activation_variants(act):
    ffn = FeedForwardNetwork(d_model=32, d_ff=128, activation=act, dropout=0.0)
    x = torch.randn(2, 4, 32)
    out = ffn(x)
    assert out.shape == (2, 4, 32)


# 4. Gradient Propagation Test
def test_ffn_gradient_propagation(sample_ffn):
    x = torch.randn(2, 4, 64, requires_grad=True)
    out = sample_ffn(x)
    loss = out.sum()
    loss.backward()

    assert sample_ffn.w_1.weight.grad is not None
    assert sample_ffn.w_2.weight.grad is not None
    assert (sample_ffn.w_1.weight.grad != 0.0).any()
    assert (sample_ffn.w_2.weight.grad != 0.0).any()


def test_swiglu_gradient_propagation():
    ffn = FeedForwardNetwork(d_model=32, d_ff=128, activation="swiglu")
    x = torch.randn(2, 4, 32, requires_grad=True)
    out = ffn(x)
    loss = out.sum()
    loss.backward()

    assert ffn.w_gate.weight.grad is not None
    assert ffn.w_up.weight.grad is not None
    assert ffn.w_down.weight.grad is not None


# 5. Weight Initializer Strategies Test
@pytest.mark.parametrize(
    "init_strategy",
    ["normal", "uniform", "xavier_uniform", "xavier_normal", "kaiming_uniform", "truncated_normal"],
)
def test_weight_initialization_strategies(init_strategy):
    cfg = FeedForwardConfig(d_model=32, d_ff=128, initializer=init_strategy)
    ffn = FeedForwardNetwork(config=cfg)
    assert ffn.d_model == 32


# 6. Validator NaN / Inf Detection & Dimension Tests
def test_validator_nan_inf_detection():
    validator = FeedForwardValidator(d_model=32)

    # Valid input
    valid_x = torch.randn(2, 4, 32)
    assert validator.validate_input_embeddings(valid_x).is_valid is True

    # NaN detection
    nan_x = torch.randn(2, 4, 32)
    nan_x[0, 1, 5] = float("nan")
    res_nan = validator.validate_input_embeddings(nan_x)
    assert res_nan.is_valid is False
    assert res_nan.has_nan is True


def test_ffn_raises_on_invalid_dimension(sample_ffn):
    bad_x = torch.randn(2, 4, 16)  # d_model mismatch (16 vs 64)
    with pytest.raises(FeedForwardValidationError):
        sample_ffn(bad_x)


# 7. Feed Forward Statistics Test
def test_ffn_statistics(sample_ffn):
    x = torch.randn(2, 4, 64)
    out = sample_ffn(x)
    out.sum().backward()

    stats = FeedForwardStatistics.compute_stats(sample_ffn)
    assert stats.d_model == 64
    assert stats.hidden_dim == 256
    assert stats.max_activation >= 0.0
    assert stats.w1_grad_norm > 0.0


# 8. Factory Creation Test
def test_ffn_factory_creation():
    app_cfg = AppConfig()
    ffn_module = FeedForwardFactory.create_feed_forward(app_cfg)
    assert isinstance(ffn_module, FeedForwardNetwork)
    assert ffn_module.d_model == app_cfg.model.d_model


# 9. Variable Sequence Lengths & Batches Test
@pytest.mark.parametrize("b_size", [1, 4])
@pytest.mark.parametrize("seq_len", [1, 16, 64])
def test_variable_sequence_lengths_and_batches(sample_ffn, b_size, seq_len):
    x = torch.randn(b_size, seq_len, 64)
    out = sample_ffn(x)
    assert out.shape == (b_size, seq_len, 64)
