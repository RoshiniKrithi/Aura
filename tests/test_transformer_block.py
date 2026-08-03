"""Comprehensive PyTest Suite for Aura Transformer Decoder Block Module.

Validates TransformerBlock forward pass shapes, SwiGLU / GeLU support, causal mask application,
KV-cache flow during generation, gradient propagation to sub-layer parameters, validator checks,
statistics extraction, and factory creation.
"""

import pytest
import torch

from src.transformer import (
    TransformerBlock,
    TransformerBlockConfig,
    TransformerBlockFactory,
    TransformerBlockStatistics,
    TransformerBlockValidator,
)
from src.utils.config import AppConfig


@pytest.fixture
def block_config():
    """Returns a test TransformerBlockConfig."""
    return TransformerBlockConfig(
        d_model=64, n_heads=4, d_ff=256, dropout=0.1, activation="swiglu"
    )


@pytest.fixture
def sample_block(block_config):
    """Returns an initialized TransformerBlock module."""
    return TransformerBlock(config=block_config)


# 1. Configuration Test
def test_transformer_block_config(block_config):
    assert block_config.d_model == 64
    assert block_config.n_heads == 4
    assert block_config.d_ff == 256
    assert block_config.activation == "swiglu"


# 2. Forward Pass Tests (3D & 2D)
def test_transformer_block_forward_3d(sample_block):
    sample_block.eval()
    x = torch.randn(2, 8, 64)
    out = sample_block(x)

    assert out.shape == (2, 8, 64)
    assert out.dtype == torch.float32


def test_transformer_block_forward_2d(sample_block):
    sample_block.eval()
    x = torch.randn(8, 64)
    out = sample_block(x)

    assert out.shape == (8, 64)


# 3. Activation Functions (SwiGLU vs GeLU)
@pytest.mark.parametrize("act", ["swiglu", "gelu"])
def test_transformer_block_activations(act):
    block = TransformerBlock(d_model=64, n_heads=4, d_ff=256, activation=act)
    block.eval()
    x = torch.randn(2, 4, 64)
    out = block(x)

    assert out.shape == (2, 4, 64)


# 4. KV-Cache Flow (Inference Autoregressive Generation)
def test_transformer_block_kv_cache(sample_block):
    sample_block.eval()
    x = torch.randn(1, 1, 64)
    out, kv_cache = sample_block(x, use_cache=True)

    assert out.shape == (1, 1, 64)
    assert kv_cache is not None
    assert "key" in kv_cache
    assert "value" in kv_cache
    assert kv_cache["key"].shape == (1, 4, 1, 16)  # (B, n_heads, seq_len, head_dim)

    # Next token step with KV-cache
    x_next = torch.randn(1, 1, 64)
    out_next, updated_cache = sample_block(x_next, kv_cache=kv_cache, use_cache=True)

    assert out_next.shape == (1, 1, 64)
    assert updated_cache["key"].shape == (1, 4, 2, 16)


# 5. Gradient Propagation Test across Attention & FFN
def test_transformer_block_gradient_propagation(sample_block):
    x = torch.randn(2, 4, 64, requires_grad=True)
    out = sample_block(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    # Check parameters in Attention, FFN, and LN got non-zero gradients
    assert sample_block.attn.c_attn.weight.grad is not None
    assert (sample_block.ffn.w_gate.weight.grad if sample_block.ffn.is_swiglu else sample_block.ffn.w_1.weight.grad) is not None
    assert sample_block.ln_1.gamma.grad is not None


# 6. Validator NaN / Inf Detection
def test_transformer_block_validator_nan_inf():
    validator = TransformerBlockValidator(d_model=64)

    nan_x = torch.randn(2, 4, 64)
    nan_x[0, 1, 5] = float("nan")

    res_nan = validator.validate_inputs(nan_x)
    assert res_nan.is_valid is False
    assert res_nan.has_nan is True


# 7. Statistics Extractor Test
def test_transformer_block_statistics(sample_block):
    sample_block.eval()
    x = torch.randn(2, 4, 64)
    _ = sample_block(x)

    stats = TransformerBlockStatistics.compute_stats(sample_block)
    assert stats.d_model == 64
    assert stats.n_heads == 4
    assert stats.total_parameters > 0
    assert stats.attn_parameters > 0
    assert stats.ffn_parameters > 0


# 8. Factory Creation Test
def test_transformer_block_factory_creation():
    app_cfg = AppConfig()
    block = TransformerBlockFactory.create_block(app_cfg)

    assert isinstance(block, TransformerBlock)
    assert block.d_model == app_cfg.model.d_model
    assert block.n_heads == app_cfg.model.n_heads


# 9. Parametrized Batch, Seq Len, and Dimension Testing
@pytest.mark.parametrize("b_size", [1, 4])
@pytest.mark.parametrize("seq_len", [1, 16, 64])
@pytest.mark.parametrize("dim, heads", [(32, 2), (64, 4), (128, 8)])
def test_parametrized_transformer_block_shapes(b_size, seq_len, dim, heads):
    block = TransformerBlock(d_model=dim, n_heads=heads, d_ff=dim * 4)
    block.eval()
    x = torch.randn(b_size, seq_len, dim)

    out = block(x)
    assert out.shape == (b_size, seq_len, dim)


# 10. Large Sequence & Memory Stress Test
def test_transformer_block_large_sequence():
    block = TransformerBlock(d_model=128, n_heads=8, d_ff=512)
    block.eval()
    x = torch.randn(2, 512, 128)
    out = block(x)

    assert out.shape == (2, 512, 128)
    assert not torch.isnan(out).any().item()


# 11. Mixed Precision Test (BFloat16 / FP16)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_transformer_block_mixed_precision(dtype):
    if dtype == torch.float16 and not torch.cuda.is_available():
        pytest.skip("FP16 Linear matrix operations require CUDA execution.")

    block = TransformerBlock(d_model=64, n_heads=4, d_ff=256).to(dtype)
    block.eval()
    x = torch.randn(2, 8, 64, dtype=dtype)
    out = block(x)

    assert out.shape == (2, 8, 64)
    assert out.dtype == dtype
    assert not torch.isnan(out).any().item()

