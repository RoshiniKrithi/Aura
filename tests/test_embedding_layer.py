"""Comprehensive PyTest Suite for Aura Token Embedding Module.

Validates weight initializers, forward pass shapes, gradient propagation, input validator
boundary checks, statistics, manager lifecycle operations, vector utilities, and factory creation.
"""

from pathlib import Path
import pytest
import torch
import torch.nn as nn

from src.embeddings import (
    EmbeddingConfig,
    EmbeddingFactory,
    EmbeddingInitializer,
    EmbeddingLayer,
    EmbeddingManager,
    EmbeddingStatistics,
    EmbeddingUtilities,
    EmbeddingValidationError,
    EmbeddingValidator,
)
from src.utils.config import AppConfig


@pytest.fixture
def default_config():
    """Returns a test EmbeddingConfig."""
    return EmbeddingConfig(vocab_size=100, d_model=32, initializer="normal", init_std=0.02)


@pytest.fixture
def sample_embedding_layer(default_config):
    """Returns an initialized EmbeddingLayer."""
    return EmbeddingLayer(config=default_config)


# 1. Configuration Tests
def test_embedding_config():
    cfg = EmbeddingConfig(vocab_size=500, d_model=128, initializer="xavier_uniform")
    assert cfg.vocab_size == 500
    assert cfg.d_model == 128
    assert cfg.initializer == "xavier_uniform"


# 2. Initializer Tests
@pytest.mark.parametrize(
    "method",
    ["normal", "uniform", "xavier_uniform", "xavier_normal", "kaiming_uniform", "truncated_normal"],
)
def test_embedding_initializers(method):
    param = nn.Parameter(torch.empty(50, 16))
    EmbeddingInitializer.initialize(param, method=method, init_range=0.05, init_std=0.02, pad_idx=0)
    assert not torch.isnan(param).any()
    assert torch.all(param[0] == 0.0)  # pad_idx vector zeroed


# 3. EmbeddingLayer Forward Pass Tests
def test_embedding_layer_forward_2d(sample_embedding_layer):
    input_ids = torch.tensor([[1, 5, 10, 99], [0, 2, 4, 8]], dtype=torch.long)
    out = sample_embedding_layer(input_ids)
    assert out.shape == (2, 4, 32)
    assert out.dtype == torch.float32


def test_embedding_layer_forward_1d(sample_embedding_layer):
    input_ids = torch.tensor([1, 5, 10], dtype=torch.long)
    out = sample_embedding_layer(input_ids)
    assert out.shape == (3, 32)


def test_embedding_layer_scaling():
    cfg = EmbeddingConfig(vocab_size=50, d_model=16, scale_by_sqrt_d_model=True)
    layer = EmbeddingLayer(config=cfg)
    input_ids = torch.tensor([[1, 2]], dtype=torch.long)
    out = layer(input_ids)

    # Manual unscaled lookup
    raw_lookup = torch.nn.functional.embedding(input_ids, layer.weight)
    assert torch.allclose(out, raw_lookup * 4.0)  # sqrt(16) = 4.0


# 4. Input Validation & Error Handling Tests
def test_validator_out_of_bounds():
    validator = EmbeddingValidator(vocab_size=100, d_model=32)

    # Negative index
    bad_neg = torch.tensor([[-1, 5]], dtype=torch.long)
    res_neg = validator.validate_input_ids(bad_neg)
    assert res_neg.is_valid is False
    assert any("negative" in err for err in res_neg.errors)

    # Exceeding vocabulary bound
    bad_oob = torch.tensor([[100, 5]], dtype=torch.long)
    res_oob = validator.validate_input_ids(bad_oob)
    assert res_oob.is_valid is False
    assert any("exceeding" in err for err in res_oob.errors)


def test_validator_invalid_dtype():
    validator = EmbeddingValidator(vocab_size=100, d_model=32)
    float_input = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    res = validator.validate_input_ids(float_input)
    assert res.is_valid is False


def test_embedding_layer_raises_on_invalid_input(sample_embedding_layer):
    bad_input = torch.tensor([[200]], dtype=torch.long)
    with pytest.raises(EmbeddingValidationError):
        sample_embedding_layer(bad_input)


# 5. Gradient Propagation Test
def test_gradient_propagation(sample_embedding_layer):
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    embeddings = sample_embedding_layer(input_ids)
    loss = embeddings.sum()
    loss.backward()

    assert sample_embedding_layer.weight.grad is not None
    # Verify non-zero gradient for accessed tokens
    assert (sample_embedding_layer.weight.grad[1] != 0.0).any()
    assert (sample_embedding_layer.weight.grad[2] != 0.0).any()
    # Verify zero gradient for unaccessed tokens
    assert (sample_embedding_layer.weight.grad[50] == 0.0).all()


# 6. EmbeddingStatistics Tests
def test_embedding_statistics(sample_embedding_layer):
    input_ids = torch.tensor([[1, 2]], dtype=torch.long)
    out = sample_embedding_layer(input_ids)
    out.sum().backward()

    stats = EmbeddingStatistics.compute_stats(sample_embedding_layer)
    assert stats.vocab_size == 100
    assert stats.d_model == 32
    assert stats.mean_vector_l2_norm > 0.0
    assert stats.grad_norm > 0.0


# 7. EmbeddingUtilities & Cosine Similarity Tests
def test_embedding_utilities(sample_embedding_layer):
    sim = EmbeddingUtilities.get_token_similarity(sample_embedding_layer, 5, 10)
    assert -1.0 <= sim <= 1.0

    neighbors = EmbeddingUtilities.top_k_nearest_neighbors(
        sample_embedding_layer, query=5, top_k=3, exclude_self=True
    )
    assert len(neighbors) == 3
    for neighbor_id, score in neighbors:
        assert neighbor_id != 5
        assert -1.0 <= score <= 1.0


# 8. EmbeddingManager Lifecycle & Weight Tying Tests
def test_embedding_manager_freezing(sample_embedding_layer):
    manager = EmbeddingManager(sample_embedding_layer)
    manager.freeze()
    assert sample_embedding_layer.weight.requires_grad is False
    manager.unfreeze()
    assert sample_embedding_layer.weight.requires_grad is True


def test_embedding_manager_weight_tying(sample_embedding_layer):
    linear = nn.Linear(32, 100, bias=False)
    manager = EmbeddingManager(sample_embedding_layer)
    manager.tie_weights(linear)

    # Verify weight tensors share exact same memory pointer
    assert linear.weight.data_ptr() == sample_embedding_layer.weight.data_ptr()


def test_embedding_manager_save_load(sample_embedding_layer, tmp_path):
    save_path = tmp_path / "embedding_weights.pt"
    manager = EmbeddingManager(sample_embedding_layer)
    manager.save_weights(save_path)
    assert save_path.exists()

    new_layer = EmbeddingLayer(vocab_size=100, d_model=32)
    new_manager = EmbeddingManager(new_layer)
    new_manager.load_weights(save_path)

    assert torch.equal(sample_embedding_layer.weight, new_layer.weight)


# 9. EmbeddingFactory Tests
def test_embedding_factory():
    app_cfg = AppConfig()
    layer = EmbeddingFactory.create_embedding_layer(app_cfg)
    assert layer.vocab_size == app_cfg.model.vocab_size
    assert layer.d_model == app_cfg.model.d_model
