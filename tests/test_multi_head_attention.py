"""Comprehensive PyTest Suite for Aura Multi-Head Causal Self-Attention Module.

Validates MultiHeadAttention forward pass shapes, head splitting, 4D causal masking across all heads,
gradient propagation, divisibility constraints, output projection, statistics, factory API,
and model scaling configurations (125M to 7B parameter specs).
"""

import pytest
import torch
import torch.nn as nn

from src.attention import (
    AttentionFactory,
    AttentionHead,
    AttentionMask,
    AttentionOutputProjection,
    AttentionStatistics,
    AttentionValidationError,
    AttentionValidator,
    MultiHeadAttention,
    MultiHeadAttentionConfig,
)
from src.utils.config import AppConfig


@pytest.fixture
def mha_config():
    """Returns a test MultiHeadAttentionConfig."""
    return MultiHeadAttentionConfig(d_model=64, n_heads=4, dropout=0.0, causal=True)


@pytest.fixture
def sample_mha(mha_config):
    """Returns an initialized MultiHeadAttention module."""
    return MultiHeadAttention(config=mha_config)


# 1. Configuration & Divisibility Tests
def test_multi_head_config(mha_config):
    assert mha_config.d_model == 64
    assert mha_config.n_heads == 4
    assert mha_config.head_dim == 16


def test_mha_divisibility_validation():
    # Should raise ValueError if d_model is not divisible by n_heads (e.g. 64 % 5 != 0)
    with pytest.raises(ValueError):
        _ = MultiHeadAttentionConfig(d_model=64, n_heads=5).head_dim

    with pytest.raises(ValueError):
        _ = MultiHeadAttention(d_model=64, n_heads=5)

    with pytest.raises(AttentionValidationError):
        AttentionValidator.validate_multi_head_config(d_model=64, n_heads=5)


# 2. Forward Pass Tests (3D & 2D)
def test_mha_forward_3d(sample_mha):
    x = torch.randn(2, 8, 64)
    out = sample_mha(x)
    assert out.shape == (2, 8, 64)
    assert out.dtype == torch.float32


def test_mha_forward_2d(sample_mha):
    x = torch.randn(8, 64)
    out = sample_mha(x)
    assert out.shape == (8, 64)


# 3. 4D Causal Masking Verification Across All Heads
def test_causal_masking_zeroes_future_across_all_heads(sample_mha):
    x = torch.randn(2, 4, 64)
    _ = sample_mha(x)

    attn_weights = sample_mha.last_attention_weights  # Shape: (B=2, H=4, T=4, T=4)
    assert attn_weights is not None
    assert attn_weights.shape == (2, 4, 4, 4)

    # Check upper triangular entries (j > i) are 0.0 across EVERY head
    for b in range(2):
        for h in range(4):
            for i in range(4):
                for j in range(i + 1, 4):
                    assert torch.abs(attn_weights[b, h, i, j]) < 1e-6


# 4. Gradient Propagation Test
def test_mha_gradient_propagation(sample_mha):
    x = torch.randn(2, 4, 64, requires_grad=True)
    out = sample_mha(x)
    loss = out.sum()
    loss.backward()

    # Verify non-zero gradients propagate to c_attn and c_proj matrices
    assert sample_mha.c_attn.weight.grad is not None
    assert sample_mha.c_proj.weight.grad is not None
    assert (sample_mha.c_attn.weight.grad != 0.0).any()
    assert (sample_mha.c_proj.weight.grad != 0.0).any()


# 5. Attention Head Abstraction Test
def test_attention_head_extraction(sample_mha):
    x = torch.randn(2, 4, 64)
    _ = sample_mha(x)

    head_0_attn = AttentionHead.extract_head_attention(
        sample_mha.last_attention_weights, head_index=0
    )
    assert head_0_attn.shape == (2, 4, 4)

    with pytest.raises(ValueError):
        _ = AttentionHead.extract_head_attention(
            sample_mha.last_attention_weights, head_index=10
        )


# 6. Attention Output Projection Test
def test_attention_output_projection():
    out_proj = AttentionOutputProjection(d_model=64, dropout=0.1)
    x = torch.randn(2, 8, 64)
    y = out_proj(x)
    assert y.shape == (2, 8, 64)


# 7. Attention Factory Multi-Head API Test
def test_mha_factory_creation():
    app_cfg = AppConfig()
    mha_module = AttentionFactory.create_multi_head_attention(app_cfg)
    assert isinstance(mha_module, MultiHeadAttention)
    assert mha_module.d_model == app_cfg.model.d_model
    assert mha_module.n_heads == app_cfg.attention.n_heads


# 8. Multi-Head Attention Statistics Test
def test_mha_statistics(sample_mha):
    x = torch.randn(2, 4, 64)
    out = sample_mha(x)
    out.sum().backward()

    stats = AttentionStatistics.compute_stats(sample_mha)
    assert stats.d_model == 64
    assert stats.d_attn == 16  # head_dim
    assert stats.mean_entropy >= 0.0
    assert stats.q_grad_norm > 0.0


# 9. Model Architecture Scaling Configurations Test
@pytest.mark.parametrize(
    "d_model, n_heads",
    [
        (768, 12),    # GPT-2 Small / 125M Spec
        (1024, 16),   # GPT-2 Medium / 350M Spec
        (2048, 16),   # GPT-2 Large / 1.3B Spec
        (4096, 32),   # LLaMA-7B / 7B Spec
    ],
)
def test_scaling_model_architectures(d_model, n_heads):
    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=0.0)
    x = torch.randn(1, 4, d_model)
    out = mha(x)
    assert out.shape == (1, 4, d_model)


# 10. Variable Sequence Lengths & Batches Test
@pytest.mark.parametrize("b_size", [1, 4])
@pytest.mark.parametrize("seq_len", [1, 16, 64])
def test_variable_sequence_lengths_and_batches(sample_mha, b_size, seq_len):
    x = torch.randn(b_size, seq_len, 64)
    out = sample_mha(x)
    assert out.shape == (b_size, seq_len, 64)
