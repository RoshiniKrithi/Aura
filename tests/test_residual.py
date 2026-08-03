"""Comprehensive PyTest Suite for Aura Residual Connection Module.

Validates ResidualConnection forward pass shapes, manual mathematical equivalence, dropout behavior,
gradient propagation, validator shape/dimension matching, statistics, and factory creation.
"""

import pytest
import torch

from src.transformer import (
    ResidualConfig,
    ResidualConnection,
    ResidualFactory,
    ResidualStatistics,
    ResidualUtilities,
    ResidualValidationError,
    ResidualValidator,
)
from src.utils.config import AppConfig


@pytest.fixture
def res_config():
    """Returns a test ResidualConfig."""
    return ResidualConfig(d_model=64, dropout=0.1, norm_position="pre_norm")


@pytest.fixture
def sample_residual(res_config):
    """Returns an initialized ResidualConnection module."""
    return ResidualConnection(config=res_config)


# 1. Configuration Tests
def test_residual_config(res_config):
    assert res_config.d_model == 64
    assert res_config.dropout == 0.1
    assert res_config.norm_position == "pre_norm"


# 2. Forward Pass Tests (3D & 2D)
def test_residual_forward_3d(sample_residual):
    sample_residual.eval()
    x = torch.randn(2, 8, 64)
    sub_out = torch.randn(2, 8, 64)
    out = sample_residual(x, sub_out)

    assert out.shape == (2, 8, 64)
    assert torch.allclose(out, x + sub_out, atol=1e-6)


def test_residual_forward_2d(sample_residual):
    sample_residual.eval()
    x = torch.randn(8, 64)
    sub_out = torch.randn(8, 64)
    out = sample_residual(x, sub_out)

    assert out.shape == (8, 64)
    assert torch.allclose(out, x + sub_out, atol=1e-6)


# 3. Manual Mathematical Equivalence Test
def test_residual_math_equivalence(sample_residual):
    sample_residual.eval()
    x = torch.randn(2, 4, 64)
    sub_out = torch.randn(2, 4, 64)

    out_module = sample_residual(x, sub_out)
    out_manual, _, _ = ResidualUtilities.apply_residual(x, sub_out, dropout_p=0.0, training=False)

    assert torch.allclose(out_module, out_manual, atol=1e-6)


# 4. Shape Mismatch Detection Test
def test_residual_shape_mismatch_raises(sample_residual):
    x = torch.randn(2, 4, 64)
    bad_sub_out = torch.randn(2, 8, 64)  # Seq len mismatch (8 vs 4)

    with pytest.raises(ResidualValidationError):
        sample_residual(x, bad_sub_out)


def test_residual_dimension_mismatch_raises():
    res = ResidualConnection(d_model=64)
    x = torch.randn(2, 4, 32)
    sub_out = torch.randn(2, 4, 32)

    with pytest.raises(ResidualValidationError):
        res(x, sub_out)


# 5. Gradient Propagation Test
def test_residual_gradient_propagation(sample_residual):
    x = torch.randn(2, 4, 64, requires_grad=True)
    sub_out = torch.randn(2, 4, 64, requires_grad=True)

    out = sample_residual(x, sub_out)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert sub_out.grad is not None
    assert torch.allclose(x.grad, torch.ones_like(x))


# 6. Validator NaN / Inf Detection
def test_residual_validator_nan_inf():
    validator = ResidualValidator(d_model=64, dropout=0.1)

    x = torch.randn(2, 4, 64)
    nan_sub = torch.randn(2, 4, 64)
    nan_sub[0, 1, 5] = float("nan")

    res_nan = validator.validate_tensors(x, nan_sub)
    assert res_nan.is_valid is False
    assert res_nan.has_nan is True


# 7. Statistics Extractor Test
def test_residual_statistics(sample_residual):
    sample_residual.eval()
    x = torch.randn(2, 4, 64)
    sub_out = torch.randn(2, 4, 64)

    _ = sample_residual(x, sub_out)
    stats = ResidualStatistics.compute_stats(sample_residual)

    assert stats.d_model == 64
    assert stats.input_norm > 0.0
    assert stats.output_norm > 0.0


# 8. Factory Creation Test
def test_residual_factory_creation():
    app_cfg = AppConfig()
    res_module = ResidualFactory.create_residual(app_cfg)

    assert isinstance(res_module, ResidualConnection)
    assert res_module.d_model == app_cfg.model.d_model


# 9. Parametrized Batch, Seq Len, and Dimension Testing
@pytest.mark.parametrize("b_size", [1, 4])
@pytest.mark.parametrize("seq_len", [1, 16, 64])
@pytest.mark.parametrize("dim", [32, 768, 4096])
def test_parametrized_residual_shapes(b_size, seq_len, dim):
    res = ResidualConnection(d_model=dim)
    res.eval()
    x = torch.randn(b_size, seq_len, dim)
    sub_out = torch.randn(b_size, seq_len, dim)

    out = res(x, sub_out)
    assert out.shape == (b_size, seq_len, dim)
