"""Comprehensive PyTest Suite for Aura Positional Embedding Module.

Validates LearnablePositionEmbedding, SinusoidalPositionEmbedding, PositionEmbeddingValidator,
PositionEmbeddingUtilities, PositionEmbeddingManager, PositionEmbeddingFactory, and InputEmbeddingPipeline.
"""

from pathlib import Path
import pytest
import torch
import torch.nn as nn

from src.embeddings import (
    EmbeddingConfig,
    EmbeddingLayer,
    EmbeddingValidationError,
    InputEmbeddingPipeline,
    LearnablePositionEmbedding,
    PositionEmbeddingConfig,
    PositionEmbeddingFactory,
    PositionEmbeddingInitializer,
    PositionEmbeddingManager,
    PositionEmbeddingUtilities,
    PositionEmbeddingValidator,
    SinusoidalPositionEmbedding,
)
from src.utils.config import AppConfig


@pytest.fixture
def pos_config():
    """Returns a test PositionEmbeddingConfig."""
    return PositionEmbeddingConfig(max_sequence_length=128, d_model=32, embedding_type="learnable")


# 1. LearnablePositionEmbedding Tests
def test_learnable_position_forward_with_seq_len(pos_config):
    module = LearnablePositionEmbedding(config=pos_config)
    out = module(sequence_length=16, batch_size=4)
    assert out.shape == (4, 16, 32)
    assert out.dtype == torch.float32


def test_learnable_position_forward_with_pos_ids(pos_config):
    module = LearnablePositionEmbedding(config=pos_config)
    pos_ids = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
    out = module(position_ids=pos_ids)
    assert out.shape == (2, 3, 32)


# 2. SinusoidalPositionEmbedding Tests
def test_sinusoidal_position_forward():
    cfg = PositionEmbeddingConfig(max_sequence_length=128, d_model=32, embedding_type="sinusoidal")
    module = SinusoidalPositionEmbedding(config=cfg)
    out = module(sequence_length=16, batch_size=2)
    assert out.shape == (2, 16, 32)

    # Verify non-trainable buffer
    assert not hasattr(module, "weight") or not module.pe.requires_grad


# 3. Sequence Length Overflow & Validation Tests
def test_sequence_length_overflow(pos_config):
    module = LearnablePositionEmbedding(config=pos_config)
    with pytest.raises(EmbeddingValidationError):
        module(sequence_length=200)  # Max is 128


def test_position_validator():
    validator = PositionEmbeddingValidator(max_sequence_length=100, d_model=32)

    res_valid = validator.validate_sequence_length(50)
    assert res_valid.is_valid is True

    res_overflow = validator.validate_sequence_length(150)
    assert res_overflow.is_valid is False

    bad_neg = torch.tensor([[-1, 0]], dtype=torch.long)
    res_neg = validator.validate_position_ids(bad_neg)
    assert res_neg.is_valid is False


# 4. Position Utilities Tests
def test_position_utilities():
    input_ids = torch.randint(0, 100, (4, 10))
    pos_ids = PositionEmbeddingUtilities.generate_position_ids(input_ids)
    assert pos_ids.shape == (4, 10)
    assert torch.equal(pos_ids[0], torch.arange(0, 10))

    module = SinusoidalPositionEmbedding(max_sequence_length=50, d_model=16)
    sim = PositionEmbeddingUtilities.compute_positional_cosine_similarity(module, 0, 1)
    assert -1.0 <= sim <= 1.0


# 5. Position Manager & Combined Embeddings Tests
def test_position_manager_combine():
    tok_emb = torch.randn(2, 5, 16)
    pos_emb = torch.randn(2, 5, 16)

    combined = PositionEmbeddingManager.combine_embeddings(tok_emb, pos_emb)
    assert combined.shape == (2, 5, 16)
    assert torch.allclose(combined, tok_emb + pos_emb)


def test_position_manager_save_load(tmp_path, pos_config):
    module = LearnablePositionEmbedding(config=pos_config)
    manager = PositionEmbeddingManager(module)

    save_path = tmp_path / "pos_weights.pt"
    manager.save_weights(save_path)
    assert save_path.exists()

    new_module = LearnablePositionEmbedding(config=pos_config)
    new_manager = PositionEmbeddingManager(new_module)
    new_manager.load_weights(save_path)

    assert torch.equal(module.weight, new_module.weight)


# 6. Position Factory Tests
def test_position_factory():
    cfg_learnable = PositionEmbeddingConfig(embedding_type="learnable")
    mod_learnable = PositionEmbeddingFactory.create_position_embedding(cfg_learnable)
    assert isinstance(mod_learnable, LearnablePositionEmbedding)

    cfg_sin = PositionEmbeddingConfig(embedding_type="sinusoidal")
    mod_sin = PositionEmbeddingFactory.create_position_embedding(cfg_sin)
    assert isinstance(mod_sin, SinusoidalPositionEmbedding)


# 7. InputEmbeddingPipeline Tests (Token + Position + Dropout)
def test_input_embedding_pipeline():
    tok_emb = EmbeddingLayer(vocab_size=100, d_model=32)
    pos_emb = LearnablePositionEmbedding(max_sequence_length=128, d_model=32)

    pipeline = InputEmbeddingPipeline(
        token_embedding=tok_emb, position_embedding=pos_emb, dropout=0.0
    )

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    out = pipeline(input_ids)
    assert out.shape == (1, 4, 32)

    # Verify Gradient Flow to both token and position weights
    loss = out.sum()
    loss.backward()

    assert tok_emb.weight.grad is not None
    assert pos_emb.weight.grad is not None


def test_input_embedding_pipeline_from_config():
    app_cfg = AppConfig()
    pipeline = InputEmbeddingPipeline.from_config(app_cfg)
    input_ids = torch.randint(0, app_cfg.model.vocab_size, (2, 16))
    out = pipeline(input_ids)
    assert out.shape == (2, 16, app_cfg.model.d_model)
