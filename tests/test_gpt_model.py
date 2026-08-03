"""Comprehensive PyTest Suite for AuraGPT Decoder LLM Architecture.

Validates end-to-end forward pass logits shapes, Cross-Entropy loss computation,
weight tying memory sharing, KV-cache generation flow across N blocks, gradient propagation,
validator checks, parameter statistics, factory creation, and scaling presets (125M to 70B).
"""

import pytest
import torch

from src.models import (
    AuraGPT,
    AuraGPTConfig,
    AuraGPTFactory,
    GPTConfig,
    GPTModel,
    ModelFactory,
    ModelStatistics,
    ModelUtilities,
    ModelValidationError,
    ModelValidator,
    TransformerStack,
)
from src.utils.config import AppConfig


@pytest.fixture
def small_gpt_config():
    """Returns a lightweight AuraGPTConfig for fast unit testing."""
    return AuraGPTConfig(
        model_name="aura-gpt-test",
        vocab_size=1000,
        max_sequence_length=512,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=256,
        dropout=0.1,
        tie_weights=True,
    )


@pytest.fixture
def sample_gpt(small_gpt_config):
    """Returns an initialized lightweight AuraGPT model."""
    return AuraGPT(config=small_gpt_config)


# 1. Configuration & Preset Tests
def test_auragpt_config_presets():
    cfg_125m = AuraGPTConfig.get_125m_config()
    assert cfg_125m.d_model == 768
    assert cfg_125m.n_layers == 12

    cfg_350m = AuraGPTConfig.get_350m_config()
    assert cfg_350m.d_model == 1024
    assert cfg_350m.n_layers == 24

    cfg_1_3b = AuraGPTConfig.get_1_3b_config()
    assert cfg_1_3b.d_model == 2048
    assert cfg_1_3b.n_layers == 24


# 2. Forward Pass Tests (2D & 1D)
def test_auragpt_forward_2d(sample_gpt):
    sample_gpt.eval()
    input_ids = torch.randint(0, 1000, (2, 8))
    logits = sample_gpt(input_ids)

    assert logits.shape == (2, 8, 1000)
    assert logits.dtype == torch.float32


def test_auragpt_forward_1d(sample_gpt):
    sample_gpt.eval()
    input_ids = torch.randint(0, 1000, (8,))
    logits = sample_gpt(input_ids)

    assert logits.shape == (8, 1000)


# 3. Training Mode Cross-Entropy Loss Computation
def test_auragpt_loss_computation(sample_gpt):
    sample_gpt.train()
    input_ids = torch.randint(0, 1000, (2, 8))
    targets = torch.randint(0, 1000, (2, 8))

    logits, loss = sample_gpt(input_ids, targets=targets)

    assert logits.shape == (2, 8, 1000)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar loss
    assert loss.item() > 0.0


# 4. Weight Tying Memory Sharing Test
def test_auragpt_weight_tying(small_gpt_config):
    model = AuraGPT(config=small_gpt_config)
    assert model.tie_weights is True
    # Verify exact parameter object memory sharing
    assert model.lm_head.weight is model.tok_embeddings.weight


# 5. KV-Cache Flow (Multi-Step Inference Generation)
def test_auragpt_kv_cache(sample_gpt):
    sample_gpt.eval()
    input_ids = torch.randint(0, 1000, (1, 1))

    logits, kv_caches = sample_gpt(input_ids, use_cache=True)
    assert logits.shape == (1, 1, 1000)
    assert len(kv_caches) == sample_gpt.n_layers
    assert "key" in kv_caches[0]

    # Step 2: Next token generation with KV-cache
    next_input_ids = torch.randint(0, 1000, (1, 1))
    logits_next, updated_caches = sample_gpt(
        next_input_ids, kv_caches=kv_caches, use_cache=True
    )

    assert logits_next.shape == (1, 1, 1000)
    assert updated_caches[0]["key"].shape == (1, 4, 2, 16)  # (B, n_heads, seq_len=2, head_dim=16)


# 6. Gradient Propagation across Full Model Architecture
def test_auragpt_gradient_propagation(sample_gpt):
    sample_gpt.train()
    input_ids = torch.randint(0, 1000, (2, 4))
    targets = torch.randint(0, 1000, (2, 4))

    logits, loss = sample_gpt(input_ids, targets=targets)
    loss.backward()

    # Check embedding, transformer block, and LM head gradients
    assert sample_gpt.tok_embeddings.weight.grad is not None
    assert sample_gpt.blocks[0].attn.c_attn.weight.grad is not None
    assert sample_gpt.ln_f.gamma.grad is not None


# 7. Validator Vocabulary & Out-of-Bounds Detection
def test_auragpt_validator_bounds():
    validator = ModelValidator(vocab_size=1000, max_sequence_length=512)

    valid_ids = torch.randint(0, 1000, (2, 8))
    assert validator.validate_inputs(valid_ids).is_valid is True

    out_of_bounds_ids = torch.tensor([[10, 1500, 20]])  # ID 1500 >= vocab_size 1000
    res_bad = validator.validate_inputs(out_of_bounds_ids)
    assert res_bad.is_valid is False


def test_auragpt_raises_on_out_of_bounds_ids(sample_gpt):
    bad_ids = torch.tensor([[10, 2000, 30]])
    with pytest.raises(ModelValidationError):
        sample_gpt(bad_ids)


# 8. Model Statistics Extractor Test
def test_auragpt_statistics(sample_gpt):
    stats = ModelStatistics.compute_stats(sample_gpt)

    assert stats.total_parameters > 0
    assert stats.n_layers == 2
    assert stats.d_model == 64
    assert stats.tie_weights is True


# 9. Factory Creation Test
def test_auragpt_factory_creation():
    app_cfg = AppConfig()
    model = AuraGPTFactory.create_model(app_cfg)

    assert isinstance(model, AuraGPT)
    assert model.d_model == app_cfg.model.d_model
    assert model.n_layers == app_cfg.model.n_layers

    # Factory preset creation test
    model_preset = AuraGPTFactory.create_model(preset="125m")
    assert model_preset.d_model == 768
    assert model_preset.n_layers == 12


# 10. Parametrized Batch, Sequence Length, and Model Tier Test
@pytest.mark.parametrize("b_size", [1, 4])
@pytest.mark.parametrize("seq_len", [1, 16, 64])
def test_parametrized_gpt_forward_shapes(b_size, seq_len):
    model = AuraGPT(config=AuraGPTConfig(vocab_size=500, d_model=32, n_layers=2, n_heads=2, d_ff=128))
    model.eval()

    input_ids = torch.randint(0, 500, (b_size, seq_len))
    logits = model(input_ids)

    assert logits.shape == (b_size, seq_len, 500)


# 11. GPTModel Decoder Trunk Tests (Hidden States Only)
def test_gpt_model_decoder_trunk_forward():
    cfg = GPTConfig(vocab_size=500, d_model=32, n_layers=2, n_heads=2, d_ff=128)
    model = GPTModel(config=cfg)
    model.eval()

    input_ids = torch.randint(0, 500, (2, 16))
    hidden_states = model(input_ids)

    # Must return hidden states only of shape (B, T, d_model)
    assert hidden_states.shape == (2, 16, 32)
    assert isinstance(hidden_states, torch.Tensor)


def test_transformer_stack_composition():
    cfg = GPTConfig(vocab_size=500, d_model=32, n_layers=2, n_heads=2, d_ff=128)
    stack = TransformerStack(config=cfg)
    stack.eval()

    x = torch.randn(2, 16, 32)
    out = stack(x)

    assert out.shape == (2, 16, 32)


def test_model_factory_create_gpt_model():
    model = ModelFactory.create_gpt_model(preset="125m")
    assert isinstance(model, GPTModel)
    assert model.d_model == 768
    assert model.n_layers == 12


def test_model_utilities_metrics():
    x = torch.randn(2, 16, 32)
    mean_val, var_val, l2_norm = ModelUtilities.compute_hidden_state_metrics(x)
    assert isinstance(mean_val, float)
    assert isinstance(var_val, float)
    assert isinstance(l2_norm, float)

