"""Comprehensive PyTest Suite for Aura Single-Head Causal Self-Attention Module.

Validates SelfAttention forward pass shapes, causal mask correctness, gradient propagation,
scaling factors, input validator NaN/Inf checks, statistics, utilities, and factory creation.
"""

from pathlib import Path
import pytest
import torch
import torch.nn as nn

from src.attention import (
    AttentionConfig,
    AttentionFactory,
    AttentionMask,
    AttentionStatistics,
    AttentionUtilities,
    AttentionValidationError,
    AttentionValidator,
    SelfAttention,
)
from src.utils.config import AppConfig


@pytest.fixture
def attn_config():
    """Returns a test AttentionConfig."""
    return AttentionConfig(d_model=32, d_attn=32, dropout=0.0, causal=True)


@pytest.fixture
def sample_attention(attn_config):
    """Returns an initialized SelfAttention module."""
    return SelfAttention(config=attn_config)


# 1. Configuration Tests
def test_attention_config():
    cfg = AttentionConfig(d_model=64, d_attn=64, causal=True, dropout=0.1)
    assert cfg.d_model == 64
    assert cfg.d_attn == 64
    assert cfg.causal is True
    assert cfg.dropout == 0.1


# 2. Causal Attention Mask Tests
def test_causal_mask_creation():
    mask = AttentionMask.create_causal_mask(sequence_length=4)
    assert mask.shape == (4, 4)
    # Allowed positions (j <= i) are 0.0
    assert mask[0, 0] == 0.0
    assert mask[1, 0] == 0.0
    assert mask[1, 1] == 0.0
    # Forbidden future positions (j > i) are -inf
    assert mask[0, 1] == float("-inf")
    assert mask[0, 2] == float("-inf")
    assert mask[1, 3] == float("-inf")


# 3. SelfAttention Forward Pass Tests
def test_self_attention_forward_3d(sample_attention):
    x = torch.randn(2, 8, 32)
    out = sample_attention(x)
    assert out.shape == (2, 8, 32)
    assert out.dtype == torch.float32


def test_self_attention_forward_2d(sample_attention):
    x = torch.randn(8, 32)
    out = sample_attention(x)
    assert out.shape == (8, 32)


# 4. Causal Masking Verification Test (No future leakage)
def test_causal_mask_zeroes_future(sample_attention):
    x = torch.randn(1, 4, 32)
    _ = sample_attention(x)

    attn_weights = sample_attention.last_attention_weights  # Shape: (1, 4, 4)
    assert attn_weights is not None

    # Check upper triangular entries (j > i) are strictly 0.0
    for i in range(4):
        for j in range(i + 1, 4):
            assert torch.abs(attn_weights[0, i, j]) < 1e-6
            assert torch.allclose(attn_weights[0, i, j], torch.tensor(0.0), atol=1e-5)


# 5. Gradient Propagation Test
def test_gradient_propagation(sample_attention):
    x = torch.randn(2, 4, 32, requires_grad=True)
    out = sample_attention(x)
    loss = out.sum()
    loss.backward()

    # Verify non-zero gradients propagate to Q, K, V, Out projections
    assert sample_attention.q_proj.weight.grad is not None
    assert sample_attention.k_proj.weight.grad is not None
    assert sample_attention.v_proj.weight.grad is not None
    assert sample_attention.out_proj.weight.grad is not None
    assert (sample_attention.q_proj.weight.grad != 0.0).any()
    assert (sample_attention.k_proj.weight.grad != 0.0).any()


# 6. Validator NaN / Inf Detection & Bounds Tests
def test_validator_nan_inf_detection():
    validator = AttentionValidator(d_model=32)

    # Valid input
    valid_x = torch.randn(2, 4, 32)
    res_valid = validator.validate_input_embeddings(valid_x)
    assert res_valid.is_valid is True

    # NaN detection
    nan_x = torch.randn(2, 4, 32)
    nan_x[0, 1, 5] = float("nan")
    res_nan = validator.validate_input_embeddings(nan_x)
    assert res_nan.is_valid is False
    assert res_nan.has_nan is True

    # Inf detection
    inf_x = torch.randn(2, 4, 32)
    inf_x[1, 2, 10] = float("inf")
    res_inf = validator.validate_input_embeddings(inf_x)
    assert res_inf.is_valid is False
    assert res_inf.has_inf is True


def test_self_attention_raises_on_invalid_input(sample_attention):
    bad_x = torch.randn(2, 4, 16)  # d_model mismatch (16 vs 32)
    with pytest.raises(AttentionValidationError):
        sample_attention(bad_x)


# 7. Attention Statistics Tests
def test_attention_statistics(sample_attention):
    x = torch.randn(2, 4, 32)
    out = sample_attention(x)
    out.sum().backward()

    stats = AttentionStatistics.compute_stats(sample_attention)
    assert stats.d_model == 32
    assert stats.mean_entropy >= 0.0
    assert stats.q_grad_norm > 0.0


# 8. Attention Utilities Tests
def test_attention_utilities():
    q = torch.randn(2, 4, 16)
    k = torch.randn(2, 4, 16)
    v = torch.randn(2, 4, 16)

    out, weights = AttentionUtilities.compute_scaled_dot_product_attention(
        q, k, v, is_causal=True
    )

    assert out.shape == (2, 4, 16)
    assert weights.shape == (2, 4, 4)


# 9. Attention Factory Tests
def test_attention_factory():
    app_cfg = AppConfig()
    module = AttentionFactory.create_self_attention(app_cfg)
    assert isinstance(module, SelfAttention)
    assert module.d_model == app_cfg.model.d_model


# 10. Variable Sequence Lengths & Batches Test
@pytest.mark.parametrize("b_size", [1, 4, 8])
@pytest.mark.parametrize("seq_len", [1, 16, 64])
def test_variable_sequence_lengths_and_batches(sample_attention, b_size, seq_len):
    x = torch.randn(b_size, seq_len, 32)
    out = sample_attention(x)
    assert out.shape == (b_size, seq_len, 32)
