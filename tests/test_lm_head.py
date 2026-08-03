"""Comprehensive PyTest Suite for Phase 14 Language Modeling Head Subsystem.

Validates logits output shapes, weight tying parameter memory sharing, untied weights,
gradient propagation, validator NaN/Inf checks, statistics extraction, and factory builders.
"""

import pytest
import torch
import torch.nn as nn

from src.models import (
    GPTConfig,
    LanguageModelHead,
    LanguageModelingHead,
    LMHeadConfig,
    LMHeadFactory,
    LMHeadStatistics,
    LMHeadUtilities,
    LMHeadValidationError,
    LMHeadValidator,
)
from src.utils.config import AppConfig


@pytest.fixture
def small_lm_head_config():
    """Returns a lightweight LMHeadConfig for fast unit testing."""
    return LMHeadConfig(d_model=64, vocab_size=1000, tie_weights=True)


@pytest.fixture
def sample_tied_weight():
    """Returns an embedding weight parameter for weight tying tests."""
    param = nn.Parameter(torch.empty(1000, 64))
    nn.init.normal_(param, mean=0.0, std=0.02)
    return param


@pytest.fixture
def sample_lm_head(small_lm_head_config, sample_tied_weight):
    """Returns an initialized LanguageModelingHead with tied weight."""
    return LanguageModelingHead(config=small_lm_head_config, tied_weight=sample_tied_weight)


# 1. Configuration Schema Tests
def test_lm_head_config_defaults():
    cfg = LMHeadConfig()
    assert cfg.d_model == 768
    assert cfg.vocab_size == 50257
    assert cfg.tie_weights is True
    assert cfg.bias is False


# 2. Output Logit Shape Tests (3D & 2D)
def test_lm_head_forward_3d(sample_lm_head):
    sample_lm_head.eval()
    hidden_states = torch.randn(2, 8, 64)
    logits = sample_lm_head(hidden_states)

    assert logits.shape == (2, 8, 1000)
    assert logits.dtype == torch.float32


def test_lm_head_forward_2d(sample_lm_head):
    sample_lm_head.eval()
    hidden_states = torch.randn(8, 64)
    logits = sample_lm_head(hidden_states)

    assert logits.shape == (8, 1000)


# 3. Weight Tying Memory Sharing Verification
def test_lm_head_weight_tying(small_lm_head_config, sample_tied_weight):
    lm_head = LanguageModelingHead(config=small_lm_head_config, tied_weight=sample_tied_weight)
    assert lm_head.tie_weights is True
    assert lm_head.weight is sample_tied_weight


# 4. Untied Weight Initialization & Forward
def test_lm_head_untied_weights():
    cfg = LMHeadConfig(d_model=32, vocab_size=500, tie_weights=False, bias=True)
    lm_head = LanguageModelingHead(config=cfg)

    assert lm_head.tie_weights is False
    assert lm_head.bias is not None
    assert lm_head.weight.shape == (500, 32)

    hidden = torch.randn(2, 4, 32)
    logits = lm_head(hidden)
    assert logits.shape == (2, 4, 500)


# 5. Gradient Propagation Test
def test_lm_head_gradient_propagation(sample_lm_head, sample_tied_weight):
    hidden = torch.randn(2, 4, 64, requires_grad=True)
    logits = sample_lm_head(hidden)

    loss = logits.sum()
    loss.backward()

    assert hidden.grad is not None
    assert sample_tied_weight.grad is not None


# 6. Validator NaN & Inf Checks
def test_lm_head_validator_nan_inf():
    validator = LMHeadValidator(d_model=64, vocab_size=1000)

    nan_tensor = torch.randn(2, 4, 64)
    nan_tensor[0, 0, 0] = float("nan")

    res_nan = validator.validate_hidden_states(nan_tensor)
    assert res_nan.is_valid is False
    assert res_nan.has_nan is True

    inf_tensor = torch.randn(2, 4, 64)
    inf_tensor[0, 0, 0] = float("inf")

    res_inf = validator.validate_hidden_states(inf_tensor)
    assert res_inf.is_valid is False
    assert res_inf.has_inf is True


def test_lm_head_raises_on_invalid_input(sample_lm_head):
    bad_hidden = torch.randn(2, 4, 128)  # d_model mismatch (128 vs 64)
    with pytest.raises(LMHeadValidationError):
        sample_lm_head(bad_hidden)


# 7. Factory Builder Tests
def test_lm_head_factory_creation():
    gpt_cfg = GPTConfig(d_model=128, vocab_size=2000)
    lm_head = LMHeadFactory.create_lm_head(gpt_cfg)

    assert isinstance(lm_head, LanguageModelingHead)
    assert lm_head.d_model == 128
    assert lm_head.vocab_size == 2000


# 8. Statistics Extractor Test
def test_lm_head_statistics(sample_lm_head):
    stats = LMHeadStatistics.compute_stats(sample_lm_head)
    assert stats.d_model == 64
    assert stats.vocab_size == 1000
    assert stats.tie_weights is True

    logits = torch.randn(2, 8, 1000)
    mean_v, var_v, l2_n = LMHeadStatistics.compute_logit_metrics(logits)
    assert isinstance(mean_v, float)
    assert isinstance(var_v, float)
    assert isinstance(l2_n, float)


# 9. Parametrized Hidden Dimensions & Vocabulary Sizes
@pytest.mark.parametrize("d_model", [32, 128])
@pytest.mark.parametrize("vocab_size", [256, 1024])
def test_parametrized_lm_head_shapes(d_model, vocab_size):
    cfg = LMHeadConfig(d_model=d_model, vocab_size=vocab_size, tie_weights=False)
    lm_head = LanguageModelingHead(config=cfg)
    lm_head.eval()

    hidden = torch.randn(4, 16, d_model)
    logits = lm_head(hidden)

    assert logits.shape == (4, 16, vocab_size)


# 10. LMHeadUtilities & Class Alias Tests
def test_lm_head_utilities():
    logits = torch.randn(2, 8, 500)
    mean_v, var_v, l2_n, min_v, max_v = LMHeadUtilities.compute_logit_stats(logits)

    assert isinstance(mean_v, float)
    assert isinstance(var_v, float)

    top_v, top_idx = LMHeadUtilities.inspect_top_k_logits(logits, k=5)
    assert top_v.shape == (2, 5)
    assert top_idx.shape == (2, 5)


def test_language_model_head_alias():
    cfg = LMHeadConfig(d_model=32, vocab_size=100, tie_weights=False)
    lm_head = LanguageModelHead(config=cfg)

    assert isinstance(lm_head, LanguageModelingHead)

