"""Comprehensive PyTest Suite for Aura Layer Normalization Module.

Validates LayerNormalization forward pass shapes, manual mathematical equivalence, zero variance stability,
gradient propagation to gamma and beta, validator NaN/Inf detection, statistics, and factory creation.
"""

import pytest
import torch
import torch.nn as nn

from src.normalization import (
    LayerNormConfig,
    LayerNormFactory,
    LayerNormalization,
    LayerNormStatistics,
    LayerNormUtilities,
    LayerNormValidationError,
    LayerNormValidator,
)
from src.utils.config import AppConfig


@pytest.fixture
def ln_config():
    """Returns a test LayerNormConfig."""
    return LayerNormConfig(d_model=64, eps=1e-5, elementwise_affine=True)


@pytest.fixture
def sample_layernorm(ln_config):
    """Returns an initialized LayerNormalization module."""
    return LayerNormalization(config=ln_config)


# 1. Configuration Tests
def test_layer_norm_config(ln_config):
    assert ln_config.d_model == 64
    assert ln_config.eps == 1e-5
    assert ln_config.elementwise_affine is True


# 2. Forward Pass Tests (3D & 2D)
def test_layernorm_forward_3d(sample_layernorm):
    x = torch.randn(2, 8, 64)
    out = sample_layernorm(x)
    assert out.shape == (2, 8, 64)
    assert out.dtype == torch.float32


def test_layernorm_forward_2d(sample_layernorm):
    x = torch.randn(8, 64)
    out = sample_layernorm(x)
    assert out.shape == (8, 64)


# 3. Manual Mathematical Equivalence Test
def test_manual_math_equivalence(sample_layernorm):
    x = torch.randn(2, 4, 64)
    out_module = sample_layernorm(x)

    # Manual Calculation: y = gamma * ((x - mean) / sqrt(var + eps)) + beta
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_hat = (x - mean) / torch.sqrt(var + 1e-5)
    out_manual = sample_layernorm.gamma * x_hat + sample_layernorm.beta

    assert torch.allclose(out_module, out_manual, atol=1e-6)


# 4. Zero Variance Stability Test (Constant Input Tensor)
def test_zero_variance_stability(sample_layernorm):
    # Constant input vector (variance = 0)
    x_const = torch.ones(2, 4, 64) * 5.0
    out = sample_layernorm(x_const)

    assert not torch.isnan(out).any().item()
    assert not torch.isinf(out).any().item()
    # With gamma=1 and beta=0, (5 - 5) / sqrt(0 + 1e-5) = 0.0
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-4)


# 5. Gradient Propagation Test
def test_layernorm_gradient_propagation(sample_layernorm):
    x = torch.randn(2, 4, 64, requires_grad=True)
    out = sample_layernorm(x)
    loss = out.sum()
    loss.backward()

    assert sample_layernorm.gamma.grad is not None
    assert sample_layernorm.beta.grad is not None
    assert (sample_layernorm.gamma.grad != 0.0).any()


# 6. Validator NaN / Inf & Bounds Tests
def test_validator_nan_inf_detection():
    validator = LayerNormValidator(d_model=64, eps=1e-5)

    valid_x = torch.randn(2, 4, 64)
    assert validator.validate_input_embeddings(valid_x).is_valid is True

    nan_x = torch.randn(2, 4, 64)
    nan_x[0, 1, 5] = float("nan")
    res_nan = validator.validate_input_embeddings(nan_x)
    assert res_nan.is_valid is False
    assert res_nan.has_nan is True


def test_layernorm_raises_on_invalid_dimension(sample_layernorm):
    bad_x = torch.randn(2, 4, 16)  # d_model mismatch (16 vs 64)
    with pytest.raises(LayerNormValidationError):
        sample_layernorm(bad_x)


# 7. LayerNorm Statistics Test
def test_layernorm_statistics(sample_layernorm):
    x = torch.randn(2, 4, 64)
    out = sample_layernorm(x)
    out.sum().backward()

    stats = LayerNormStatistics.compute_stats(sample_layernorm)
    assert stats.d_model == 64
    assert stats.eps == 1e-5
    assert stats.gamma_norm > 0.0


# 8. LayerNorm Utilities Test
def test_layernorm_utilities():
    x = torch.randn(2, 4, 64)
    out, mean, var = LayerNormUtilities.compute_layer_norm(x, eps=1e-5)

    assert out.shape == (2, 4, 64)
    assert mean.shape == (2, 4, 1)
    assert var.shape == (2, 4, 1)


# 9. Factory Creation Test
def test_layernorm_factory_creation():
    app_cfg = AppConfig()
    ln_module = LayerNormFactory.create_layer_norm(app_cfg)
    assert isinstance(ln_module, LayerNormalization)
    assert ln_module.d_model == app_cfg.model.d_model


# 10. Variable Sequence Lengths & Batches Test
@pytest.mark.parametrize("b_size", [1, 4])
@pytest.mark.parametrize("seq_len", [1, 16, 64])
@pytest.mark.parametrize("dim", [32, 768, 4096])
def test_variable_sequence_lengths_and_batches(b_size, seq_len, dim):
    ln = LayerNormalization(d_model=dim)
    x = torch.randn(b_size, seq_len, dim)
    out = ln(x)
    assert out.shape == (b_size, seq_len, dim)
